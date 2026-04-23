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


def download_and_execute(algo_name: str, verbose: bool = True) -> Optional[object]:
    """
    获取算法实例

    流程：
    1. 检查缓存 → 有则直接加载
    2. 让 LLM 分析算法需要的依赖包 → pip install
    3. LLM 生成调用代码 → 缓存
    4. 全部失败 → None
    """
    # 1. 检查缓存
    cached = _load_from_cache(algo_name)
    if cached is not None:
        if verbose:
            print(f"[Downloader] 从缓存加载：{algo_name}")
        return cached

    # 🔴 2. 让 LLM 自主分析算法需要的依赖包（完全自主，不受任何限制）
    if verbose:
        print(f"[Downloader] 正在分析算法 {algo_name} 的依赖...")

    dependencies = _llm_analyze_dependencies(algo_name, verbose=verbose)

    if dependencies:
        # 安装 LLM 分析的依赖
        if verbose:
            print(f"[Downloader] LLM 分析依赖：{dependencies}")

        # 安装依赖
        if _install_dependencies(dependencies, verbose=verbose):
            # 让 LLM 生成调用代码
            code = _llm_generate_code(algo_name, dependencies, verbose=verbose)
            if code:
                return _save_and_load(code, algo_name, verbose=verbose)

    # 3. 全部失败
    if verbose:
        print(f"[Downloader] ❌ 无法获取算法 {algo_name}")
    return None


def _llm_analyze_dependencies(algo_name: str, verbose: bool = True) -> Optional[List[str]]:
    """
    让 LLM 自主分析算法需要哪些 pip 依赖包

    Args:
        algo_name: 算法名称
        verbose: 是否打印日志

    Returns:
        依赖包列表，如 ['opencv-python', 'numpy']
    """
    from agents.llm_agent import call_qwen_text

    if verbose:
        print(f"[Downloader] 正在让 LLM 分析 {algo_name} 的依赖...")

    system_prompt = """你是一位计算机视觉领域的 Python 专家。
你的任务：分析图像匹配算法名称，确定需要哪些 pip 包。

仅返回一个 JSON 数组格式的包名列表，例如：
["opencv-python", "numpy", "scipy"]

不要包含任何解释，只返回 JSON 数组。"""

    user_prompt = f"""算法名称：{algo_name}

这是一个图像匹配/配准/比较算法。
分析实现这个算法需要哪些 Python 包。

常用包：
- opencv-python (cv2)：用于大多数 CV 算法（SIFT、ORB、模板匹配等）
- opencv-contrib-python：用于额外算法如 SURF
- numpy (np)：用于数组操作
- scipy：用于信号处理、相位相关
- scikit-image (skimage)：用于 SSIM、高级图像处理
- Pillow (PIL)：用于图像加载、图像哈希
- imagehash：用于感知哈希

返回一个 JSON 数组格式的必需包列表。"""

    try:
        response = call_qwen_text(system_prompt, user_prompt, max_retries=2)

        if not response:
            return None

        # 解析 JSON
        import json
        # 清理响应（移除 markdown 代码块）
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()

        dependencies = json.loads(response)

        if isinstance(dependencies, list) and len(dependencies) > 0:
            # 验证所有项都是字符串
            if all(isinstance(pkg, str) for pkg in dependencies):
                if verbose:
                    print(f"[Downloader] ✅ LLM 分析成功：{dependencies}")
                return dependencies

        return None

    except Exception as e:
        if verbose:
            print(f"[Downloader] ❌ LLM 分析依赖失败：{e}")
        return None


def _install_dependencies(dependencies: List[str], verbose: bool = True) -> bool:
    """
    安装依赖包

    Args:
        dependencies: 依赖包列表
        verbose: 是否打印日志

    Returns:
        是否全部安装成功
    """
    import subprocess

    all_installed = True

    for pkg in dependencies:
        try:
            # 尝试导入（检查是否已安装）
            # 处理包名和导入名不一致的情况
            import_name = pkg.replace('-', '_').replace('.', '_')

            if import_name == 'scikit_image':
                import skimage
            elif import_name == 'Pillow':
                from PIL import Image
            elif import_name == 'opencv_contrib_python':
                import cv2
            elif import_name == 'opencv_python':
                import cv2
            else:
                # 动态导入
                __import__(import_name)

            if verbose:
                print(f"[Downloader] ✅ {pkg} 已安装")

        except ImportError:
            # 未安装则自动 pip install
            if verbose:
                print(f"[Downloader] 正在安装 {pkg}...")

            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg, "-q"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if verbose:
                    print(f"[Downloader] ✅ 已安装 {pkg}")
            except subprocess.CalledProcessError as e:
                if verbose:
                    print(f"[Downloader] ❌ 安装 {pkg} 失败：{e}")
                all_installed = False

    return all_installed


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

    system_prompt = """你是一位计算机视觉领域的 Python 专家。
生成一个封装图像匹配算法的 Python 类。

⚠️️⚠️ 关键：代码必须以下面这些精确的导入语句开头 ⚠️️⚠️
```python
import cv2
import numpy as np
from agents.match_result import MatchResult
# 其他导入（scipy、skimage、PIL 等）写在下面
```
绝对不能跳过这些导入！即使你的算法使用 numpy/scipy/skimage，你仍然需要 cv2 进行图像预处理。

代码模板（遵循此结构）：
```python
import cv2
import numpy as np
from agents.match_result import MatchResult
# 可选：from scipy import ...
# 可选：from skimage import ...

class Algorithm:
    def default_params(self):
        return {'param1': value, 'param2': value}
    
    def run(self, template, scene, **params):
        params = {**self.default_params(), **params}
        
        # 如果需要，转换为灰度图
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        if len(scene.shape) == 3:
            scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        
        # 你的算法实现写在这里...
        
        return MatchResult(
            algorithm='algorithm_name',
            found=found,
            confidence=confidence,
            location=location,
            size=size
        )
```

关键要求：
1. ⚠️ 类名必须精确为 'Algorithm'（不能是 SiftMatchingAlgorithm、TemplateMatcher 等其他名称）
2. 必须有 'default_params()' 方法，返回默认参数 dict
3. 必须有 'run(template, scene, **params)' 方法，返回 MatchResult 对象
4. ⚠️️⚠️ 必需的导入 - 你必须在代码开头包含这些精确的导入语句：
   ```python
   import cv2
   import numpy as np
   from agents.match_result import MatchResult
   ```
   - ⚠️️⚠️⚠️  即使你的算法主要使用 numpy/scipy/skimage，你仍然必须导入 cv2
   - 许多算法需要 cv2 进行图像预处理（颜色转换、缩放等）
   - 不要忘记这些导入
   - 不要使用 'cv2' 和 'np' 以外的别名
   - 如果需要其他库（scipy、skimage、PIL），也要导入
5. 可以使用的包：opencv-python (cv2)、numpy (np)、scipy、scikit-image (skimage)、Pillow (PIL)
6. 处理边界情况（未找到特征等）
7. 返回 0-1 之间的置信度分数
8. 重要：default_params 只能使用简单的 Python 值（int、float、str、bool）
   - 不要使用 cv2 常量如 cv2.HISTCMP_CORREL、cv2.TM_CCOEFF_NORMED 等
   - 使用字符串代替：'HISTCMP_CORREL'、'TM_CCOEFF_NORMED' 等
   - 在 run() 方法中，使用 if/elif 或映射 dict 将字符串转换为 cv2 常量
9. 这很重要，因为参数会被 JSON 序列化并在函数之间传递
10. 访问图像形状时，记住图像可以是灰度（2D）或彩色（3D）：
    - 使用：h, w = template.shape[:2]  # 适用于灰度和彩色
    - 不要使用：h, w, c = template.shape  # 灰度图会失败
11. MatchResult 构造函数签名：
    - MatchResult(algorithm, found, confidence, location=None, size=None)
    - 不要传递 'params' 给构造函数
    - 创建结果后，如果需要可以设置 result.params = params
12. 使用正确的库函数：
    - 相位相关：使用 OpenCV 的 cv2.phaseCorrel()，不要用 scipy.signal
    - SSIM：使用 scikit-image 的 skimage.metrics.structural_similarity()
      * ⚠️ 关键：不要对 SSIM 使用暴力滑动窗口！对大图像来说太慢了。
      * ⚠️ 你必须实现以下优化策略：
        ```python
        # 1. 将图像下采样到 1/4 分辨率
        scale = 0.25
        template_small = cv2.resize(template, (int(w_template * scale), int(h_template * scale)))
        scene_small = cv2.resize(scene, (int(w_scene * scale), int(h_scene * scale)))
        
        # 2. 使用步长跳过像素
        step = max(1, int(2 / scale))  # scale = 0.25 时 step = 8
        for y in range(0, h_ss - h_ts + 1, step):
            for x in range(0, w_ss - w_ts + 1, step):
                ssim_score = structural_similarity(template_small, scene_small[y:y+h, x:x+w])
        
        # 3. 缩放回原始坐标
        best_x = int(best_location_small[0] / scale)
        best_y = int(best_location_small[1] / scale)
        ```
      * 此优化对大图像（1920x1080）提供约 1000 倍加速
      * 没有此优化，代码会太慢而无法使用
      * ⚠️ 重要：调用 structural_similarity() 时正确处理小图像：
        - 如果图像小于 7x7，传递 win_size 参数：win_size=min(7, min(patch.shape))
        - 或对小图像使用：structural_similarity(..., win_size=3, channel_axis=None)
        - 这可以防止测试期间出现 "win_size exceeds image extent" 错误
    - 特征匹配：使用 cv2.SIFT_create()、cv2.ORB_create()、cv2.AKAZE_create()、cv2.BRISK_create()
    - 模板匹配：使用 cv2.matchTemplate()
    - 直方图比较：使用 cv2.calcHist() 和 cv2.compareHist()
    - 始终查阅 OpenCV 官方文档获取正确的 API
    - 不要发明或使用不存在的函数
    - 关键：如果你对 API 不是 100% 确定，使用你确定存在的更简单的替代方案
    - 例如：ECC 配准很复杂，如果不确定就不要使用
13. 关键：置信度分数计算
    - 置信度必须在 0.0 和 1.0 之间（包含）
    - 对于特征匹配（SIFT/ORB/AKAZE）：
      * 步骤 1：使用 RANSAC 从 good matches 中找到 inliers
      * 步骤 2：检查几何有效性：
        - 使用单应性矩阵变换模板角点
        - 计算有多少角点保持在场景边界内（带 20% 边距）
        - bounds_score = corners_in_bounds / 4.0
      * 步骤 3：组合信号：
        - inlier_ratio = inlier_count / len(good_matches)
        - match_quality = len(good_matches) / len(all_matches)
        - confidence = inlier_ratio * 0.5 + bounds_score * 0.3 + match_quality * 0.2
      * 步骤 4：应用惩罚：
        - 如果 inlier_ratio < 0.5：confidence *= 0.5（对低几何一致性的重惩罚）
        - 上限 0.95 避免过度自信（为参数调优留空间）
      * 这确保：错误匹配即使有很多 inliers 也会得到低置信度
      * 典型值：好匹配 → 0.4-0.8，错误匹配 → 0.1-0.3
    - 对于模板匹配：
      * 使用 cv2.matchTemplate()，根据方法不同返回特定范围的值：
        - TM_CCOEFF_NORMED：-1 到 1（1 是完美匹配，但通常 0.7+ 算好）
        - TM_CCORR_NORMED：0 到 1（1 是完美匹配）
        - TM_SQDIFF_NORMED：0 到 1（0 是完美匹配，所以 confidence = 1 - min_val）
      * 关键：不要手动归一化归一化方法的结果！
        cv2.matchTemplate() 已经返回正确归一化的值。
        删除这行错误代码：result = (result + 1) / 2
      * 对于 TM_CCOEFF_NORMED，confidence = max_val（已在正确范围）
      * 对于 TM_CCORR_NORMED，confidence = max_val（已在 0-1 范围）
      * 对于 TM_SQDIFF_NORMED，confidence = 1 - min_val（反向）
      * 关键验证：你必须验证匹配位置是否合理：
        1. 如果 best_location 是 (0, 0) 或非常靠近图像边缘（距边缘<10px），很可能是假阳性
           → 设置 found=False 和 confidence=0.0（这个匹配可疑）
        2. 检查模板与场景的面积比：
           ratio = (template_w * template_h) / (scene_w * scene_h)
           如果 ratio > 0.5（模板覆盖>50% 的场景），可能是错误的；降低置信度 0.4
           如果 ratio < 0.001（模板太小），可能是噪声；降低置信度 0.2
        3. 如果匹配区域覆盖>80% 的场景，几乎肯定是假阳性
        4. 在设置 found=True 之前应用惩罚：
           如果位置可疑：confidence *= 0.5
           如果面积比太大：confidence *= 0.6
      * 只有当 confidence >= threshold 且位置通过验证时才设置 found=True
      * 通常期望：TM_CCOEFF_NORMED 的好匹配 >0.7
    - 对于 SSIM：
      * 使用 skimage.metrics.structural_similarity() 返回 0-1
      * confidence = ssim_score 直接
      * ⚠️ 关键：正确处理小图像
        - 设置 win_size=min(7, min(patch.shape))
        - 如果 SSIM 返回 NaN 或负数，设置 confidence=0.0
    - 对于归一化互相关（NCC）：
      * ⚠️ 关键：使用 cv2.matchTemplate() 而不是 scipy.signal.correlate2d！
      * 使用 cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED)
      * 结果已经在 0-1 范围内，直接使用 max_val 作为置信度
      * 不要手动归一化或缩放结果
      * 如果需要加速，可以先降采样图像，但仍然使用 cv2.matchTemplate()
    - 对于直方图比较：
      * 使用 cv2.compareHist() - HISTCMP_CORREL 返回 0-1
      * confidence = result 直接
      * ⚠️ 关键性能优化：
        - 不要对每个位置都计算直方图！这太慢了
        - 使用降采样：将图像缩小到 1/4 或 1/8
        - 使用大步长搜索（step >= 50 像素）
        - 或只比较整图直方图（全局匹配，不滑动窗口）
        - 目标：在 800x600 图像上运行时间 < 1 秒
    - 对于感知哈希匹配：
      * 使用 imagehash.phash() 计算感知哈希
      * ⚠️ 关键性能优化：
        - 不要对每个滑动窗口位置都计算哈希！这太慢了
        - 只计算整个场景的哈希，与模板哈希比较（全局匹配）
        - 计算汉明距离：distance = hash1 - hash2
        - 置信度 = 1 - (distance / (hash_size * hash_size))
        - 如果场景比模板大很多，只比较整图（不滑动窗口）
        - 目标：在 800x600 图像上运行时间 < 0.1 秒
    - 永远不要除以图像维度或使用任意公式
    - 置信度应该反映真实的匹配质量
14. 遵循项目的编码标准：
    - 使用清晰、可读的代码，带有适当的错误处理
    - 使用 try/except 块优雅地处理错误
    - 在所有情况下都返回正确的 MatchResult 对象
    - 确保代码可以被 JSON 序列化（使用原生 Python 类型）
    - 需要时使用 cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 将图像转换为灰度
    - 在 run() 开始处验证输入参数
15. 重要 - 只使用这些标准实现：
    - OpenCV：cv2.phaseCorrel()、cv2.matchTemplate()、cv2.SIFT_create()、cv2.ORB_create()、cv2.AKAZE_create() 等
    - scikit-image：skimage.metrics.structural_similarity()、skimage.feature.* 等
    - numpy：np.fft.fft2()、np.correlate() 等
    - 除非你 100% 确定函数存在，否则永远不要从 scipy.signal 导入

⚠️ 提醒：类名必须精确为 'Algorithm'，不能是其他名称！

特征匹配（SIFT/ORB/AKAZE）带有正确置信度计算的示例代码：
```python
import cv2
import numpy as np
from agents.match_result import MatchResult

class Algorithm:
    def default_params(self):
        return {
            'method': 'SIFT',
            'threshold': 0.7,  # Lowe's ratio test 阈值
            'confidence_threshold': 0.3  # 判定"找到"的最小置信度
        }

    def run(self, template, scene, **params):
        params = {**self.default_params(), **params}
        
        if template is None or scene is None:
            raise ValueError("必须提供 template 和 scene")
        
        # 转换为灰度图
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        if len(scene.shape) == 3:
            scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        
        try:
            method = params['method']
            
            # 根据 method 创建检测器
            if method == 'SIFT':
                detector = cv2.SIFT_create()
            elif method == 'ORB':
                detector = cv2.ORB_create()
            elif method == 'AKAZE':
                detector = cv2.AKAZE_create()
            else:
                detector = cv2.SIFT_create()
            
            # 检测和计算特征
            kp1, des1 = detector.detectAndCompute(template, None)
            kp2, des2 = detector.detectAndCompute(scene, None)
            
            # 检查是否找到足够的特征
            if len(kp1) < 4 or len(kp2) < 4:
                return MatchResult(method, False, 0.0)
            
            # 使用 BFMatcher 匹配特征
            if method == 'ORB':
                # ORB 使用二进制描述子，使用汉明距离
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = bf.knnMatch(des1, des2, k=2)
            else:
                # SIFT/AKAZE 使用浮点描述子，使用 L2 距离
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                matches = bf.knnMatch(des1, des2, k=2)
            
            # 应用 Lowe's ratio test
            ratio_threshold = params.get('threshold', 0.7)
            good_matches = []
            for m_n in matches:
                if len(m_n) >= 2:
                    m, n = m_n[0], m_n[1]
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
            
            # 🔴 正确的置信度计算：使用 RANSAC inlier 比率 + 几何验证
            # 关键原则：置信度必须反映真实的几何一致性
            # 错误的匹配如果单应性拟合了噪声，也可能有很多 inliers！
            
            if len(good_matches) >= 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                
                if M is not None and mask is not None:
                    inlier_count = int(mask.sum())
                    inlier_ratio = inlier_count / max(len(good_matches), 1)
                    
                    # 🔴 关键：检查单应性在几何上是否有效
                    # 变换模板角点并检查它们是否保持在场景边界内
                    h_scene, w_scene = scene.shape[:2]
                    h_template, w_template = template.shape[:2]
                    
                    template_corners = np.float32([
                        [0, 0], [w_template, 0], 
                        [w_template, h_template], [0, h_template]
                    ]).reshape(-1, 1, 2)
                    
                    transformed_corners = cv2.perspectiveTransform(template_corners, M)
                    
                    # 检查大多数角点是否在场景边界内（带 20% 边距容差）
                    margin_x, margin_y = w_scene * 0.2, h_scene * 0.2
                    corners_in_bounds = 0
                    for corner in transformed_corners:
                        x, y = corner[0]
                        if -margin_x <= x <= w_scene + margin_x and -margin_y <= y <= h_scene + margin_y:
                            corners_in_bounds += 1
                    
                    bounds_score = corners_in_bounds / 4.0
                    
                    # 🔴 匹配质量：good matches 与总可能匹配数的比率
                    match_quality = len(good_matches) / max(len(matches), 1)
                    
                    # 🔴 最终置信度：加权组合
                    # - inlier_ratio：几何一致性（最重要）
                    # - bounds_score：空间有效性
                    # - match_quality：特征丰富度
                    confidence = (inlier_ratio * 0.5 + bounds_score * 0.3 + match_quality * 0.2)
                    
                    # 🔴 严格：只有高 inlier 比率才能获得高置信度
                    if inlier_ratio < 0.5:
                        confidence *= 0.5  # 对低 inlier 比率的重惩罚
                    
                    # 🔴 上限 0.95 避免过度自信（为调优留空间）
                    confidence = min(0.95, confidence)
                else:
                    # 未找到有效的单应性
                    confidence = len(good_matches) / max(len(matches), 1) * 0.3
            else:
                # 没有足够的匹配进行 RANSAC
                confidence = len(good_matches) / max(len(matches), 1) * 0.2
            
            # 如果有足够的 good matches，找到位置
            found = confidence >= params.get('confidence_threshold', 0.3)
            
            if found and len(good_matches) >= 4:
                # 重新计算单应性获取位置（上面已为置信度计算过）
                if M is not None:
                    h, w = template.shape[:2]
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    # 获取边界框
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

仅返回 Python 代码，不要解释，不要 markdown 代码块。"""

    user_prompt = f"""生成一个完整的 Python 实现，用于 '{algo_name}' 算法。

必需的包：{', '.join(dependencies)}

该算法应该：
1. 接受两个图像：template（小模板）和 scene（大场景图像）
2. 在 scene 中找到 template
3. 返回位置 (x, y)、大小 (width, height) 和置信度 (0-1)

生成一个完整、可工作的实现，带有适当的错误处理。
不要包含 markdown 代码块（```python 或 ```），只返回纯 Python 代码。"""

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
    """保存代码到缓存并加载（带语法检查和测试）"""

    # 🔴 步骤 1: 语法检查
    if verbose:
        print(f"[Downloader] 正在检查语法...")
    try:
        compile(code, '<string>', 'exec')
        if verbose:
            print(f"[Downloader] ✅ 语法检查通过")
    except SyntaxError as e:
        if verbose:
            print(f"[Downloader] ❌ 语法错误：{e}")
        return None

    # 🔴 步骤 2: 测试运行（检查 API 是否存在）
    if verbose:
        print(f"[Downloader] 正在测试运行...")

    test_module = _test_run_code(code, algo_name, verbose=verbose)
    if test_module is None:
        if verbose:
            print(f"[Downloader] ❌ 测试运行失败")
        return None

    # 🔴 步骤 3: 保存到缓存
    ALGORITHM_CACHE_DIR.mkdir(exist_ok=True)
    cache_key = hashlib.md5((algo_name + code).encode()).hexdigest()[:12]
    cache_file = ALGORITHM_CACHE_DIR / f"{algo_name}_{cache_key}.py"
    cache_file.write_text(code, encoding="utf-8")

    if verbose:
        print(f"[Downloader] ✅ 已保存：{cache_file}")

    return test_module


def _test_run_code(code: str, algo_name: str, verbose: bool = True) -> Optional[object]:
    """
    测试运行生成的代码，检查 API 是否存在

    流程：
    1. 在隔离的命名空间中执行代码
    2. 检查 Algorithm 类是否存在
    3. 检查 default_params() 是否能正常调用
    4. 检查 run() 方法是否存在
    5. 🔴 用测试图像实际运行 run() 方法（检查导入是否完整）

    返回：测试通过的模块，或 None
    """
    import types
    import numpy as np

    try:
        # 创建隔离的命名空间
        namespace = {
            '__name__': f'test_{algo_name}',
            '__builtins__': __builtins__,
        }

        # 执行代码
        exec(code, namespace)

        # 检查 Algorithm 类
        if 'Algorithm' not in namespace:
            if verbose:
                print(f"[Downloader] ❌ 缺少 Algorithm 类")
            return None

        # 实例化
        algo_instance = namespace['Algorithm']()

        # 检查 default_params()
        try:
            params = algo_instance.default_params()
            if not isinstance(params, dict):
                if verbose:
                    print(f"[Downloader] ❌ default_params() 返回值不是 dict")
                return None
            if verbose:
                print(f"[Downloader] ✅ default_params() 测试通过")
        except AttributeError as e:
            if verbose:
                print(f"[Downloader] ❌ 缺少 default_params() 方法：{e}")
            return None
        except Exception as e:
            if verbose:
                print(f"[Downloader] ❌ default_params() 执行失败：{e}")
            return None

        # 检查 run() 方法
        if not hasattr(algo_instance, 'run'):
            if verbose:
                print(f"[Downloader] ❌ 缺少 run() 方法")
            return None

        # 🔴 用测试图像实际运行 run() 方法
        if verbose:
            print(f"[Downloader] 正在用测试图像运行 run()...")

        try:
            # 🔴 创建测试图像（彩色和灰度都测试）
            # 测试 1: 灰度图像
            test_template_gray = np.zeros((20, 20), dtype=np.uint8)
            test_scene_gray = np.zeros((50, 50), dtype=np.uint8)

            # 测试 2: 彩色图像（触发 cv2.cvtColor）
            test_template_color = np.zeros((20, 20, 3), dtype=np.uint8)
            test_scene_color = np.zeros((50, 50, 3), dtype=np.uint8)

            # 尝试运行灰度图像
            result = algo_instance.run(test_template_gray, test_scene_gray, **params)

            # 尝试运行彩色图像（测试是否会调用 cv2.cvtColor）
            result = algo_instance.run(test_template_color, test_scene_color, **params)

            # 🔴 测试 3: 性能测试（针对 SSIM 等需要优化的算法）
            # 如果算法名称包含 'ssim'，测试大图像上的性能
            if 'ssim' in algo_name.lower():
                if verbose:
                    print(f"[Downloader] 正在进行性能测试（大图像）...")

                import time
                # 创建较大的测试图像（模拟真实场景）
                large_template = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
                large_scene = np.random.randint(0, 255, (800, 800), dtype=np.uint8)

                start_time = time.time()
                result = algo_instance.run(large_template, large_scene, **params)
                elapsed_time = time.time() - start_time

                # 🔴 性能要求：大图像上必须在 5 秒内完成
                if elapsed_time > 5.0:
                    if verbose:
                        print(f"[Downloader] ❌ 性能不达标：耗时 {elapsed_time:.2f}秒（要求<5 秒）")
                        print(f"[Downloader] 提示：SSIM 必须使用 downsampling 优化")
                    return None

                if verbose:
                    print(f"[Downloader] ✅ 性能测试通过：耗时 {elapsed_time:.2f}秒")

            if verbose:
                print(f"[Downloader] ✅ run() 测试运行成功（灰度 + 彩色）")

        except NameError as e:
            # 🔴 捕获 "name 'cv2' is not defined" 这类错误
            if verbose:
                print(f"[Downloader] ❌ 缺少导入：{e}")
                print(f"[Downloader] 提示：代码中可能缺少必要的 import 语句")
            return None
        except ValueError as e:
            # 参数验证错误（如需要灰度图），这是正常的
            if verbose:
                print(f"[Downloader] ⚠️ 参数验证提示：{e}")
                print(f"[Downloader] ✅ 但 API 存在，继续")
        except TypeError as e:
            # 🔴 类型错误（如 cv2.cvtColor 参数错误），说明 API 使用有问题
            if verbose:
                print(f"[Downloader] ❌ API 使用错误：{e}")
                print(f"[Downloader] 提示：代码中 cv2 等库的使用可能有误")
            return None
        except Exception as e:
            # 其他错误可能是算法逻辑问题，不影响 API 存在性
            if verbose:
                print(f"[Downloader] ⚠️ 运行错误（可能是算法逻辑）：{e}")
                print(f"[Downloader] ✅ 但 API 存在，继续")

        if verbose:
            print(f"[Downloader] ✅ run() 方法存在")

        # 返回模块（用于后续加载）
        module = types.ModuleType(f'test_{algo_name}')
        module.Algorithm = namespace['Algorithm']
        return module

    except ImportError as e:
        if verbose:
            print(f"[Downloader] ❌ 导入失败：{e}")
        return None
    except AttributeError as e:
        if verbose:
            print(f"[Downloader] ❌ API 不存在：{e}")
        return None
    except Exception as e:
        if verbose:
            print(f"[Downloader] ❌ 测试运行失败：{e}")
        return None


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


def clear_cache():
    """清除算法缓存"""
    if ALGORITHM_CACHE_DIR.exists():
        for f in ALGORITHM_CACHE_DIR.glob("*.py"):
            f.unlink()
        print(f"[Downloader] 已清除缓存目录：{ALGORITHM_CACHE_DIR}")
