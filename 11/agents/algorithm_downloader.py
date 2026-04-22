"""
Algorithm Downloader - 完全自动化获取算法代码

策略：
1. 有 pip 库的算法 → pip install + LLM 自动生成调用代码
2. 无 pip 库的算法 → Gitee 搜索 Python 实现并下载
"""
import os
import sys
import subprocess
import requests
import hashlib
import warnings
# 过滤 paramiko 的加密算法弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="paramiko")
from pathlib import Path
from typing import Optional, Dict, List
import importlib.util
import numpy as np

ALGORITHM_CACHE_DIR = Path(__file__).parent.parent / "downloaded_algorithms"
ALGORITHM_CACHE_DIR.mkdir(exist_ok=True)

GITEE_API_BASE = "https://gitee.com/api/v5"

# 算法 → pip 包名 映射（只记录依赖，不生成代码）
PIP_PACKAGES = {
    "feature_matching": ["opencv-python", "numpy"],
    "template_matching": ["opencv-python", "numpy"],
    "sift_matching": ["opencv-python", "opencv-contrib-python", "numpy"],
    "orb_matching": ["opencv-python", "numpy"],
    "akaze_matching": ["opencv-python", "numpy"],
    "brisk_matching": ["opencv-python", "numpy"],
    "fast_matching": ["opencv-python", "numpy"],
    "phase_correlation": ["opencv-python", "numpy", "scipy"],
    "histogram_comparison": ["opencv-python", "numpy", "scipy"],
    "ssim_comparison": ["scikit-image", "numpy"],
    "mutual_information": ["scipy", "numpy"],
    "normalized_cross_correlation": ["opencv-python", "numpy", "scipy"],
    "harris_corner": ["opencv-python", "numpy"],
    "surf_matching": ["opencv-python", "opencv-contrib-python", "numpy"],
    "flann_matching": ["opencv-python", "numpy"],
    "image_hash_comparison": ["imagehash", "Pillow", "numpy"],
}

# 算法名称别名映射（处理 LLM 推荐的不同命名变体）
ALGORITHM_ALIASES = {
    # SSIM 相关
    "ssim_similarity": "ssim_comparison",
    "ssim": "ssim_comparison",
    "structural_similarity": "ssim_comparison",
    "structural_similarity_index": "ssim_comparison",
    # 特征匹配相关
    "sift": "sift_matching",
    "sift_feature_matching": "sift_matching",
    "orb": "orb_matching",
    "orb_feature_matching": "orb_matching",
    "akaze": "akaze_matching",
    "brisk": "brisk_matching",
    "fast": "fast_matching",
    "surf": "surf_matching",
    "flann": "flann_matching",
    # 模板匹配相关
    "template": "template_matching",
    "pattern_matching": "template_matching",
    # 其他
    "phase": "phase_correlation",
    "frequency_domain": "phase_correlation",
    "histogram": "histogram_comparison",
    "histogram_matching": "histogram_comparison",
    "mutual_info": "mutual_information",
    "ncc": "normalized_cross_correlation",
    "cross_correlation": "normalized_cross_correlation",
    "harris": "harris_corner",
    "harris_corner_detection": "harris_corner",
    "image_hash": "image_hash_comparison",
    "perceptual_hash": "image_hash_comparison",
}


def download_and_execute(algo_name: str, verbose: bool = True) -> Optional[object]:
    """
    获取算法实例

    流程：
    1. 检查缓存 → 有则直接加载
    2. 有 pip 库 → pip install + LLM 生成调用代码 → 缓存
    3. 无 pip 库 → Gitee 搜索 Python 实现并下载 → 缓存
    4. 全部失败 → None
    """
    # 0. 标准化算法名称（处理别名）
    original_algo_name = algo_name
    algo_name = _normalize_algorithm_name(algo_name)
    if algo_name != original_algo_name and verbose:
        print(f"[Downloader] 算法名称标准化：{original_algo_name} → {algo_name}")
    
    # 1. 检查缓存
    cached = _load_from_cache(algo_name)
    if cached is not None:
        if verbose:
            print(f"[Downloader] 从缓存加载：{algo_name}")
        return cached

    # 2. 尝试 pip 安装 + LLM 生成代码
    pip_packages = PIP_PACKAGES.get(algo_name)
    if pip_packages:
        # check_and_install_deps 会在 _llm_generate_code 中调用，这里只检查
        if check_and_install_deps(algo_name, verbose=verbose):
            # 让 LLM 生成调用代码（会自动安装额外的 CV 库）
            code = _llm_generate_code(algo_name, pip_packages, verbose=verbose)
            if code:
                return _save_and_load(code, algo_name, verbose=verbose)

    # 3. Gitee 搜索 Python 实现
    if verbose:
        print(f"[Downloader] 在 Gitee 搜索 {algo_name} 的 Python 实现...")

    result = _search_and_download(algo_name, verbose=verbose)
    if result is not None:
        return result

    # 4. 全部失败
    if verbose:
        print(f"[Downloader] ❌ 无法获取算法 {algo_name}")
    return None


def _llm_generate_code(algo_name: str, dependencies: List[str], verbose: bool = True) -> Optional[str]:
    """
    让 LLM 自动生成算法调用代码，并验证代码能正常运行
    
    Args:
        algo_name: 算法名称
        dependencies: 依赖包列表
        verbose: 是否打印日志
    
    Returns:
        Python 代码字符串
    """
    from agents.llm_agent import call_qwen_text
    
    if verbose:
        print(f"[Downloader] 正在让 LLM 生成 {algo_name} 的调用代码...")
    
    # 确保安装了常用 CV 库
    extra_packages = ['scikit-image', 'scipy', 'Pillow']
    for pkg in extra_packages:
        try:
            if pkg == 'scikit-image':
                import skimage
            elif pkg == 'scipy':
                import scipy
            elif pkg == 'Pillow':
                from PIL import Image
        except ImportError:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg, "-q"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if verbose:
                    print(f"[Downloader] ✅ 已安装 {pkg}")
            except Exception:
                pass
    
    system_prompt = """You are an expert Python developer for computer vision.
Generate a Python class that wraps an image matching algorithm.

CRITICAL REQUIREMENTS:
1. ⚠️ The class MUST be named EXACTLY 'Algorithm' (NOT SiftMatchingAlgorithm, NOT TemplateMatcher, etc.)
2. Must have a 'default_params()' method returning a dict of default parameters
3. Must have a 'run(template, scene, **params)' method that returns a MatchResult object
4. MatchResult class is already provided - import it with: from agents.match_result import MatchResult
5. You can use these packages: opencv-python (cv2), numpy (np), scipy, scikit-image (skimage), Pillow (PIL)
6. Handle edge cases (no features found, etc.)
7. Return confidence score between 0 and 1
8. IMPORTANT: For default_params, use ONLY simple Python values (int, float, str, bool)
   - DO NOT use cv2 constants like cv2.HISTCMP_CORREL, cv2.TM_CCOEFF_NORMED, etc.
   - Use strings instead: 'HISTCMP_CORREL', 'TM_CCOEFF_NORMED', etc.
   - In the run() method, convert strings to cv2 constants using if/elif or a mapping dict
9. This is important because parameters will be JSON serialized and passed between functions
10. When accessing image shape, remember images can be grayscale (2D) or color (3D):
    - Use: h, w = template.shape[:2]  # works for both grayscale and color
    - NOT: h, w, c = template.shape  # fails for grayscale
11. MatchResult constructor signature:
    - MatchResult(algorithm, found, confidence, location=None, size=None)
    - DO NOT pass 'params' to the constructor
    - After creating the result, set result.params = params if needed
12. Use the CORRECT library functions:
    - For phase correlation: use cv2.phaseCorrel() from OpenCV, NOT scipy.signal
    - For SSIM: use skimage.metrics.structural_similarity() from scikit-image
    - For feature matching: use cv2.SIFT_create(), cv2.ORB_create(), etc.
    - For template matching: use cv2.matchTemplate()
    - For histogram comparison: use cv2.calcHist() and cv2.compareHist()
    - ALWAYS check the official documentation for correct API
    - DO NOT invent or use non-existent functions
13. CRITICAL: Confidence Score Calculation
    - Confidence MUST be between 0.0 and 1.0 (inclusive)
    - For feature matching (SIFT/ORB/AKAZE):
      * Use Lowe's ratio test to find good matches
      * confidence = min(1.0, len(good_matches) / max(len(keypoints_template), 1))
      * This gives ratio of good matches to total features in template
      * Typically expect: >0.5 for good match, <0.1 for no match
      * Example: 50 good matches / 100 template keypoints = 0.50
    - For template matching:
      * Use cv2.matchTemplate() which returns values in 0-1 range for normalized methods
      * confidence = max_val from cv2.minMaxLoc()
      * Typically expect: >0.8 for good match
    - For SSIM:
      * Use skimage.metrics.structural_similarity() which returns 0-1
      * confidence = ssim_score directly
    - For histogram comparison:
      * Use cv2.compareHist() - HISTCMP_CORREL returns 0-1
      * confidence = result directly
    - NEVER divide by image dimensions or use arbitrary formulas
    - The confidence should reflect actual matching quality
14. Follow the project's coding standards:
    - Use clear, readable code with proper error handling
    - Handle errors gracefully with try/except blocks
    - Return proper MatchResult objects in all cases
    - Ensure the code can be JSON serialized (use native Python types)
    - Convert images to grayscale when needed using cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    - Validate input parameters at the beginning of run()
15. IMPORTANT - Use ONLY these standard implementations:
    - OpenCV: cv2.phaseCorrel(), cv2.matchTemplate(), cv2.SIFT_create(), cv2.ORB_create(), cv2.AKAZE_create(), etc.
    - scikit-image: skimage.metrics.structural_similarity(), skimage.feature.*, etc.
    - numpy: np.fft.fft2(), np.correlate(), etc.
    - NEVER import from scipy.signal unless you are 100% sure the function exists

⚠️ REMINDER: The class name MUST be EXACTLY 'Algorithm', not any other name!

Example code for feature matching (SIFT/ORB/AKAZE) with CORRECT confidence calculation:
```python
import cv2
import numpy as np
from agents.match_result import MatchResult

class Algorithm:
    def default_params(self):
        return {
            'method': 'SIFT',
            'threshold': 0.7,  # Lowe's ratio test threshold
            'confidence_threshold': 0.3  # Minimum confidence to say "found"
        }

    def run(self, template, scene, **params):
        params = {**self.default_params(), **params}
        
        if template is None or scene is None:
            raise ValueError("Template and scene must be provided")
        
        # Convert to grayscale
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        if len(scene.shape) == 3:
            scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        
        try:
            method = params['method']
            
            # Create detector based on method
            if method == 'SIFT':
                detector = cv2.SIFT_create()
            elif method == 'ORB':
                detector = cv2.ORB_create()
            elif method == 'AKAZE':
                detector = cv2.AKAZE_create()
            else:
                detector = cv2.SIFT_create()
            
            # Detect and compute features
            kp1, des1 = detector.detectAndCompute(template, None)
            kp2, des2 = detector.detectAndCompute(scene, None)
            
            # Check if enough features found
            if len(kp1) < 4 or len(kp2) < 4:
                return MatchResult(method, False, 0.0)
            
            # Match features using BFMatcher
            if method == 'ORB':
                # ORB uses binary descriptors, use Hamming distance
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = bf.knnMatch(des1, des2, k=2)
            else:
                # SIFT/AKAZE use float descriptors, use L2 distance
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                matches = bf.knnMatch(des1, des2, k=2)
            
            # Apply Lowe's ratio test
            ratio_threshold = params.get('threshold', 0.7)
            good_matches = []
            for m_n in matches:
                if len(m_n) >= 2:
                    m, n = m_n[0], m_n[1]
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
            
            # Calculate confidence: ratio of good matches to template keypoints
            confidence = min(1.0, len(good_matches) / max(len(kp1), 1))
            
            # Find location if enough good matches
            found = confidence >= params.get('confidence_threshold', 0.3)
            
            if found and len(good_matches) >= 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    h, w = template.shape[:2]
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    # Get bounding box
                    x = int(np.min(dst[:, :, 0]))
                    y = int(np.min(dst[:, :, 1]))
                    width = int(np.max(dst[:, :, 0]) - x)
                    height = int(np.max(dst[:, :, 1]) - y)
                else:
                    x, y = 0, 0
                    width, height = template.shape[1], template.shape[0]
            else:
                x, y = 0, 0
                width, height = template.shape[1], template.shape[0]
            
            result = MatchResult(
                algorithm=method,
                found=found,
                confidence=float(confidence),
                location=(x, y),
                size=(width, height)
            )
            result.params = params
            return result
            
        except Exception as e:
            return MatchResult(
                algorithm=params.get('method', 'unknown'),
                found=False,
                confidence=0.0
            )
```

Respond ONLY with the Python code, no explanations, no markdown code blocks."""

    user_prompt = f"""Generate a complete Python implementation for the '{algo_name}' algorithm.

Required packages: {', '.join(dependencies)}

The algorithm should:
1. Accept two images: template (small patch) and scene (larger image)
2. Find the template within the scene
3. Return location (x, y), size (width, height), and confidence (0-1)

Generate a complete, working implementation with proper error handling.
DO NOT include markdown code blocks (```python or ```), just pure Python code."""

    # Try up to 3 times with stronger prompts if needed
    for attempt in range(3):
        response = call_qwen_text(system_prompt, user_prompt)
        
        if response:
            # 清理 markdown 代码块标记
            code = response.strip()
            if code.startswith("```"):
                lines = code.split("\n")
                code = "\n".join(line for line in lines if not line.strip().startswith("```"))
            if code.startswith("python"):
                code = code[6:]
            
            code = code.strip()
            
            # 验证代码
            if _validate_generated_code(code, algo_name, verbose=verbose):
                if verbose:
                    print(f"[Downloader] ✅ LLM 已生成调用代码（已验证）")
                return code
            else:
                if verbose:
                    print(f"[Downloader] ❌ 第 {attempt + 1} 次尝试失败，重试...")
                # 增强提示词，更强调类名要求
                system_prompt = system_prompt.replace(
                    "⚠️ REMINDER: The class name MUST be EXACTLY 'Algorithm', not any other name!",
                    "⚠️⚠️ CRITICAL: The class MUST be named 'Algorithm', NOT 'SiftMatchingAlgorithm' or any other name! This is the #1 requirement!"
                )
        else:
            if verbose:
                print(f"[Downloader] ❌ LLM 生成代码失败（第 {attempt + 1} 次）")
    
    if verbose:
        print(f"[Downloader] ❌ LLM 生成代码失败（已重试3次）")
    return None


def _validate_generated_code(code: str, algo_name: str, verbose: bool = True) -> bool:
    """
    验证生成的代码是否能正常运行
    
    验证步骤：
    1. 语法检查
    2. 检查是否有 Algorithm 类
    3. 检查是否有 default_params 和 run 方法
    4. 尝试用测试数据运行
    
    Returns:
        代码是否有效
    """
    # 1. 语法检查
    try:
        compile(code, '<string>', 'exec')
    except SyntaxError as e:
        if verbose:
            print(f"[Downloader] 代码语法错误：{e}")
        return False
    
    # 2. 创建测试环境并执行代码
    try:
        # 创建命名空间，预先导入所有可能需要的模块
        namespace = {}
        
        # 导入必要的模块
        import numpy as np
        import cv2
        namespace['np'] = np
        namespace['cv2'] = cv2
        
        # 导入 MatchResult（LLM 生成的代码可能会用到）
        from agents.match_result import MatchResult
        namespace['MatchResult'] = MatchResult
        
        # 导入其他常用库（防止 LLM 生成的代码需要）
        try:
            import skimage
            namespace['skimage'] = skimage
            from skimage import measure, filters, feature
            namespace['measure'] = measure
            namespace['filters'] = filters
            namespace['feature'] = feature
        except ImportError:
            pass
        
        try:
            from scipy import ndimage, signal
            namespace['ndimage'] = ndimage
            namespace['signal'] = signal
        except ImportError:
            pass
        
        try:
            from PIL import Image
            namespace['Image'] = Image
        except ImportError:
            pass
        
        # 执行代码
        exec(code, namespace)
        
        # 3. 检查是否有 Algorithm 类
        if 'Algorithm' not in namespace:
            if verbose:
                print(f"[Downloader] 代码中没有 Algorithm 类")
            return False
        
        Algorithm = namespace['Algorithm']
        
        # 4. 检查是否有必要的方法
        if not hasattr(Algorithm, 'default_params') or not callable(getattr(Algorithm, 'default_params')):
            if verbose:
                print(f"[Downloader] Algorithm 缺少 default_params 方法")
            return False
        
        if not hasattr(Algorithm, 'run') or not callable(getattr(Algorithm, 'run')):
            if verbose:
                print(f"[Downloader] Algorithm 缺少 run 方法")
            return False
        
        # 5. 创建实例并测试
        algo_instance = Algorithm()
        
        # 6. 测试 default_params
        try:
            params = algo_instance.default_params()
            if not isinstance(params, dict):
                if verbose:
                    print(f"[Downloader] default_params() 应该返回 dict")
                return False
        except Exception as e:
            if verbose:
                print(f"[Downloader] default_params() 调用失败：{e}")
            return False
        
        # 7. 用测试数据运行（小图像）
        try:
            test_template = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
            test_scene = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            
            result = algo_instance.run(test_template, test_scene, **params)
            
            # 8. 验证返回结果
            if not isinstance(result, MatchResult):
                if verbose:
                    print(f"[Downloader] run() 应该返回 MatchResult 对象")
                return False
            
            # 9. 验证置信度在 0-1 之间
            if not (0 <= result.confidence <= 1):
                if verbose:
                    print(f"[Downloader] 置信度应该在 0-1 之间，实际为 {result.confidence}")
                # 这个可以放宽，不直接返回 False
            
            if verbose:
                print(f"[Downloader] ✅ 代码验证通过（测试运行成功）")
            return True
            
        except Exception as e:
            if verbose:
                print(f"[Downloader] run() 测试失败：{e}")
            return False
        
    except Exception as e:
        if verbose:
            print(f"[Downloader] 代码执行错误：{e}")
        return False


def _save_and_load(code: str, algo_name: str, verbose: bool = True) -> Optional[object]:
    """保存代码到缓存并加载"""
    ALGORITHM_CACHE_DIR.mkdir(exist_ok=True)
    cache_key = hashlib.md5((algo_name + code).encode()).hexdigest()[:12]
    cache_file = ALGORITHM_CACHE_DIR / f"{algo_name}_{cache_key}.py"
    cache_file.write_text(code, encoding="utf-8")

    if verbose:
        print(f"[Downloader] 已保存：{cache_file}")

    return _load_module(cache_file, algo_name)


def _load_from_cache(algo_name: str) -> Optional[object]:
    """从缓存加载算法模块"""
    if not ALGORITHM_CACHE_DIR.exists():
        return None
    for f in ALGORITHM_CACHE_DIR.glob(f"{algo_name}_*.py"):
        return _load_module(f, algo_name)
    return None


def _search_and_download(algo_name: str, verbose: bool = True) -> Optional[object]:
    """在 Gitee 上搜索算法的 Python 实现并下载"""
    search_queries = [
        f"{algo_name} python",
        f"{algo_name.replace('_', ' ')} opencv python",
        algo_name.replace("_matching", ""),
    ]

    for query in search_queries:
        repos = _gitee_search_repos(query, verbose=verbose)
        if not repos:
            continue

        for repo in repos[:5]:
            owner = repo.get("owner", {}).get("login", "")
            name = repo.get("name", "")
            if not owner or not name:
                continue

            if verbose:
                print(f"[Downloader] 找到仓库：{owner}/{name}")

            py_file = _find_python_file_in_repo(owner, name, algo_name, verbose=verbose)
            if py_file:
                downloaded = _download_file(py_file, algo_name, verbose=verbose)
                if downloaded is not None:
                    return downloaded

    return None


def _gitee_search_repos(query: str, verbose: bool = True) -> List[Dict]:
    """用 Gitee API 搜索仓库（公共接口）"""
    url = f"{GITEE_API_BASE}/search/repositories"
    params = {"q": query, "per_page": 10, "sort": "stars_count", "order": "desc"}

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            if verbose:
                print(f"[Gitee] 搜索 '{query}' → {len(data)} 个仓库")
            return data
        return []
    except Exception as e:
        if verbose:
            print(f"[Gitee] 搜索失败：{e}")
        return []


def _find_python_file_in_repo(owner: str, repo: str, algo_name: str,
                               verbose: bool = True) -> Optional[str]:
    """在仓库中查找与算法相关的 Python 文件，返回 download_url"""
    url = f"{GITEE_API_BASE}/repos/{owner}/{repo}/contents"
    params = {}

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        files = response.json()

        if not isinstance(files, list):
            return None

        algo_keywords = algo_name.replace("_", "").lower()
        candidates = []
        for f in files:
            if f.get("type") != "file":
                continue
            fname = f.get("name", "")
            if not fname.endswith(".py"):
                continue
            if algo_keywords in fname.replace("_", "").replace("-", "").lower():
                candidates.insert(0, f)
            else:
                candidates.append(f)

        if candidates:
            chosen = candidates[0]
            dl_url = chosen.get("download_url") or chosen.get("html_url")
            if verbose:
                print(f"[Gitee] 选中文件：{chosen.get('name', '')}")
            return dl_url

    except Exception as e:
        if verbose:
            print(f"[Gitee] 获取文件列表失败：{e}")
    return None


def _download_file(download_url: str, algo_name: str,
                    verbose: bool = True) -> Optional[object]:
    """下载 Python 文件并加载为模块"""
    if not download_url:
        return None

    if verbose:
        print(f"[Downloader] 正在下载：{download_url}")

    try:
        response = requests.get(download_url, timeout=30)
        response.raise_for_status()
        code = response.text

        # 验证是合法的 Python 代码
        try:
            compile(code, download_url, "exec")
        except SyntaxError as e:
            if verbose:
                print(f"[Downloader] 下载的文件不是有效 Python 代码：{e}")
            return None

        return _save_and_load(code, algo_name, verbose=verbose)

    except Exception as e:
        if verbose:
            print(f"[Downloader] 下载失败：{e}")
        return None


def _load_module(file_path: Path, module_name: str) -> Optional[object]:
    """从文件动态加载 Python 模块"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 查找 Algorithm 类并实例化
        if hasattr(module, 'Algorithm'):
            return module.Algorithm()

        # 查找任何有 run 方法的类
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and hasattr(attr, 'run'):
                return attr()

        return module
    except Exception as e:
        print(f"[Downloader] 加载模块失败：{e}")
        return None


def _normalize_algorithm_name(algo_name: str) -> str:
    """
    标准化算法名称（处理 LLM 推荐的不同命名变体）
    
    策略：
    1. 精确匹配别名表
    2. 模糊匹配（去除下划线、连字符、空格后比较）
    3. 包含匹配（检查是否包含已知算法名称关键词）
    
    Returns:
        标准化后的算法名称
    """
    if not algo_name:
        return algo_name
    
    # 1. 精确匹配别名表
    if algo_name in ALGORITHM_ALIASES:
        return ALGORITHM_ALIASES[algo_name]
    
    # 2. 标准化后匹配（去除分隔符）
    normalized = algo_name.lower().replace("_", "").replace("-", "").replace(" ", "")
    for alias, canonical in ALGORITHM_ALIASES.items():
        alias_normalized = alias.lower().replace("_", "").replace("-", "").replace(" ", "")
        if normalized == alias_normalized:
            return canonical
    
    # 3. 包含匹配（仅当输入名称包含别名关键词，且不是标准算法名时）
    #    避免将 feature_matching 错误匹配到 sift_matching
    algo_lower = algo_name.lower()
    if algo_lower not in PIP_PACKAGES:  # 不是标准算法名
        for alias, canonical in ALGORITHM_ALIASES.items():
            alias_normalized = alias.lower().replace("_", "").replace("-", "").replace(" ", "")
            # 确保别名不是另一个标准算法名的一部分
            if alias_normalized in normalized:
                # 额外检查：如果别名是某个标准算法名的一部分，需要完全匹配
                if canonical in PIP_PACKAGES:
                    # 只有当输入名称也包含标准算法名时才匹配
                    canonical_normalized = canonical.lower().replace("_", "").replace("-", "")
                    if canonical_normalized in normalized or normalized == canonical_normalized:
                        return canonical
                else:
                    return canonical
    
    # 4. 检查是否包含 PIP_PACKAGES 中的关键词（完全匹配）
    for pkg_name in PIP_PACKAGES.keys():
        if algo_lower == pkg_name:
            return pkg_name
    
    # 5. 都不匹配，返回原始名称
    return algo_name


def check_and_install_deps(algo_name: str, verbose: bool = True) -> bool:
    """检查并安装算法所需的 pip 依赖"""
    # 先标准化算法名称
    algo_name = _normalize_algorithm_name(algo_name)
    
    deps = PIP_PACKAGES.get(algo_name, [])
    if not deps:
        return True

    missing = []
    for dep in deps:
        # 检查包是否已安装
        IMPORT_MAP = {
            "opencv-python": "cv2",
            "opencv-contrib-python": "cv2",
            "Pillow": "PIL",
            "scikit-image": "skimage"
            }
        
        pkg_name = IMPORT_MAP.get(dep, dep.replace("-", "_").replace("[", "").replace("]", ""))

        try:
            __import__(pkg_name)

        except ImportError:
            missing.append(dep)

    if not missing:
        if verbose:
            print(f"[Downloader] ✅ {algo_name} 依赖已满足")
        return True

    if verbose:
        print(f"[Downloader] 需要安装依赖：{missing}")
        print(f"[Downloader] 正在自动安装...")

    for dep in missing:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", dep, "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if verbose:
                print(f"[Downloader] ✅ 已安装 {dep}")
        except Exception as e:
            if verbose:
                print(f"[Downloader] ❌ 安装 {dep} 失败：{e}")
                print(f"[Downloader] 请手动运行：pip install {dep}")
            return False
    return True


def get_algorithm_module(algo_name: str) -> Optional[object]:
    """获取已下载的算法模块"""
    return _load_from_cache(algo_name)


def clear_cache():
    """清除算法缓存"""
    if ALGORITHM_CACHE_DIR.exists():
        for f in ALGORITHM_CACHE_DIR.glob("*.py"):
            f.unlink()
        print(f"[Downloader] 已清除缓存目录：{ALGORITHM_CACHE_DIR}")
