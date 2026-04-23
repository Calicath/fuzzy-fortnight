"""
LLM Agent - 纯文本大模型分析图像特征

功能：
1. 调用 qwen-vl 分析图像（纯文本输出）
2. 推荐合适的图像匹配算法（排除深度学习）
3. 建议参数调优
"""
import os
import cv2
import json
import base64
import time
import requests
from typing import Dict, List, Optional
import config

# 全局限流控制
_last_request_time = 0
_min_request_interval = 0.5  # 最小请求间隔（秒）

def _rate_limit():
    """简单的速率限制，确保请求之间有间隔"""
    global _last_request_time
    current_time = time.time()
    elapsed = current_time - _last_request_time
    if elapsed < _min_request_interval:
        sleep_time = _min_request_interval - elapsed
        time.sleep(sleep_time)
    _last_request_time = time.time()

def extract_image_features(image: cv2.Mat) -> str:
    """
    使用 OpenCV 提取图像特征的文本描述
    """
    features = []

    features.append(f"- 分辨率：{image.shape[1]}x{image.shape[0]}")

    if len(image.shape) == 3:
        features.append("- 颜色模式：彩色")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        features.append("- 颜色模式：灰度")
        gray = image

    edges = cv2.Canny(gray, 100, 200)
    edge_density = cv2.countNonZero(edges) / (gray.shape[0] * gray.shape[1])
    features.append(f"- 边缘密度：{edge_density:.1%}")

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    high_freq_ratio = hist[128:].sum() / hist.sum()
    features.append(f"- 高频纹理占比：{high_freq_ratio:.1%}")

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    features.append(f"- 清晰度（Laplacian方差）：{sharpness:.2f}")

    return "\n".join(features)


def image_to_base64(image: cv2.Mat, quality: int = 80) -> str:
    """将 OpenCV 图像转为 base64 字符串"""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer.tobytes()).decode('utf-8')


def run_llm_recommendation(template: cv2.Mat, scene: cv2.Mat, verbose: bool = False) -> List[Dict]:
    """
    使用 LLM 推荐算法（支持视觉模型和文本模型）
    """
    # 构建提示词 - 极简，让 agent 完全自主
    system_prompt = """你是一位计算机视觉专家。

观察这些图像并推荐 4-10 个你认为最合适的匹配算法。

你有完全的自由推荐任何传统（非深度学习）的图像匹配/配准/比较算法。不要局限于常见算法——根据你观察到的图像特征创造性地思考。

重要指导原则：
1. 首先分析图像：纹理、边缘、尺度、旋转、光照、图案
2. 推荐适合图像特征的算法
3. 在算法适合的条件下可以混合不同类型：基于特征、基于模板、频域、统计、哈希等
4. 使用简短、标准的算法名称（例如："sift_matching"、"template_matching"、"phase_correlation"）
5. 不要推荐深度学习方法（不要 CNN、神经网络、SuperPoint/SuperGlue/LoFTR）

示例算法类型（非 exhaustive——发挥你的创造力）：
- 基于特征：sift_matching, orb_matching, akaze_matching, brisk_matching, surf_matching, fast_matching, harris_corner
- 基于模板：template_matching, pattern_matching, normalized_cross_correlation
- 频域：phase_correlation, fourier_transform_matching, wavelet_matching
- 统计：ssim, histogram_comparison, mutual_information, structural_similarity
- 哈希：image_hash, perceptual_hash, phash, ahash, dhash
- 其他：optical_flow, lucas_kanade, farneback, ecc_alignment, ransac_matching

对于每个算法：
- 根据你观察到的图像解释为什么选择它
- 提供合理的初始参数（使用简洁的参数字典）
- 根据你的分析给出置信度分数（0-1）

⚠️ 重要：必须返回完整、合法的 JSON 数组
- 确保所有字符串使用双引号
- 确保所有括号正确闭合
- 确保 JSON 可以被直接解析
- 不要包含任何解释文字，只返回 JSON

返回格式：
[
  {"algorithm": "sift_matching", "confidence": 0.95, "reason": "你的推理", "initial_params": {"param1": "value1"}},
  ...
]"""

    # 🔴 检测是否使用视觉模型（模型名包含 vl 或 vision）
    use_vision = any(keyword in config.QWEN_MODEL.lower() for keyword in ['vl', 'vision'])

    if use_vision:
        if verbose:
            print(f"[LLM Agent] 使用视觉模型：{config.QWEN_MODEL}，直接分析图像...")
        response_text = call_qwen_vision(system_prompt, template, scene, verbose=verbose)
    else:
        # 文本模型：提取特征描述
        if verbose:
            print("[LLM Agent] 使用文本模型，提取图像特征...")
        template_desc = extract_image_features(template)
        scene_desc = extract_image_features(scene)

        user_prompt = f"""Template Image Description:
{template_desc}

Scene Image Description:
{scene_desc}

What algorithms would you recommend for matching these images?"""

        if verbose:
            print("[LLM Agent] 调用文本模型...")

        response_text = call_qwen_text(system_prompt, user_prompt)

    if response_text is None:
        if verbose:
            print("[LLM Agent] LLM 调用失败，使用默认推荐")
        return get_default_recommendations()

    # 解析 JSON 结果
    recommendations = parse_recommendations(response_text, verbose=verbose)

    if verbose and recommendations:
        print(f"[LLM Agent] 成功推荐 {len(recommendations)} 个算法")

    return recommendations or get_default_recommendations()


def call_qwen_vision(system_prompt: str, template: cv2.Mat, scene: cv2.Mat,
                     verbose: bool = False, max_retries: int = 3) -> Optional[str]:
    """
    调用视觉模型，直接发送图像（qwen-vl 等）
    """
    url = config.QWEN_BASE_URL.rstrip("/") + "/chat/completions"

    headers = {
        "Authorization": f"Bearer {config.QWEN_API_KEY}",
        "Content-Type": "application/json",
    }

    # 将图像转为 base64
    template_b64 = image_to_base64(template)
    scene_b64 = image_to_base64(scene)

    # 🔴 压缩图像（视觉模型不需要太高分辨率）
    max_dim = 1024
    h1, w1 = template.shape[:2]
    if max(h1, w1) > max_dim:
        scale = max_dim / max(h1, w1)
        template = cv2.resize(template, (int(w1 * scale), int(h1 * scale)))
        template_b64 = image_to_base64(template, quality=70)

    h2, w2 = scene.shape[:2]
    if max(h2, w2) > max_dim:
        scale = max_dim / max(h2, w2)
        scene = cv2.resize(scene, (int(w2 * scale), int(h2 * scale)))
        scene_b64 = image_to_base64(scene, quality=70)

    # 🔴 视觉模型的消息格式（OpenAI 兼容 API）
    data = {
        "model": config.QWEN_MODEL,
        "temperature": 0.3,  # 🔴 提高温度到 0.3，增加多样性
        "top_p": 0.9,  # 🔴 添加 top_p 采样，增加输出的多样性
        "max_tokens": 6000,  # 🔴 增加到 6000，防止 JSON 被截断（推荐 4-10 个算法需要更多 tokens）
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{template_b64}"}
                    },
                    {
                        "type": "text",
                        "text": "图1：模板图（要查找的小图）"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{scene_b64}"}
                    },
                    {
                        "type": "text",
                        "text": "图2：场景图（待搜索的大图）。请分析这两张图像，推荐合适的匹配算法。"
                    }
                ]
            }
        ],
    }

    _rate_limit()

    for attempt in range(max_retries):
        try:
            if verbose:
                print(f"[LLM Agent] 第 {attempt + 1} 次调用视觉模型...")
            response = requests.post(url, json=data, headers=headers, timeout=60)

            if response.status_code == 429:
                wait_time = 2 ** (attempt + 1)
                print(f"[LLM Agent] 遇到限流（429），等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            result = response.json()

            finish_reason = result["choices"][0].get("finish_reason", "")
            content = result["choices"][0]["message"]["content"]

            if finish_reason == "length":
                print(f"[LLM Agent] ⚠️ 响应被截断，正在修复 JSON...")
                content = _repair_truncated_json(content)

            return content

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"[LLM Agent] 请求失败：{e}，{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"[LLM Agent] 请求失败（已重试{max_retries}次）：{e}")
                return None
        except Exception as e:
            print(f"[LLM Agent] 请求失败：{e}")
            return None

    return None


def call_qwen_text(system_prompt: str, user_prompt: str, max_retries: int = 3) -> Optional[str]:
    """
    调用 qwen 文本模型，带重试机制
    """
    url = config.QWEN_BASE_URL.rstrip("/") + "/chat/completions"

    headers = {
        "Authorization": f"Bearer {config.QWEN_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": config.QWEN_MODEL,
        "temperature": 0.3,  # 🔴 提高温度到 0.3，增加多样性
        "top_p": 0.9,  # 🔴 添加 top_p 采样，增加输出的多样性
        "max_tokens": 6000,  # 🔴 增加到 6000，防止 JSON 被截断（推荐 4-10 个算法需要更多 tokens）
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    # 应用速率限制
    _rate_limit()

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, headers=headers, timeout=60)  # 🔴 超时改为 60 秒

            # 处理 429 限流错误
            if response.status_code == 429:
                wait_time = 2 ** (attempt + 1)  # 指数退避: 2s, 4s, 8s
                print(f"[LLM Agent] 遇到限流（429），等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            result = response.json()

            # 🔴 检查是否被截断（finish_reason 为 length 表示 max_tokens 不够）
            finish_reason = result["choices"][0].get("finish_reason", "")
            content = result["choices"][0]["message"]["content"]

            if finish_reason == "length":
                print(f"[LLM Agent] ⚠️ 响应被截断（max_tokens 不够），正在修复 JSON...")
                # 尝试修复被截断的 JSON
                content = _repair_truncated_json(content)

            return content

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"[LLM Agent] 请求失败：{e}，{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"[LLM Agent] 请求失败（已重试{max_retries}次）：{e}")
                return None
        except Exception as e:
            print(f"[LLM Agent] 请求失败：{e}")
            return None

    return None


def _repair_truncated_json(text: str) -> str:
    """修复被截断的 JSON"""
    text = text.strip()

    # 🔴 策略 1：找到最后一个完整的对象，截断后面的
    # 从后往前找最后一个完整的 "}"
    last_complete_brace = -1
    brace_count = 0
    in_string = False
    escape_next = False

    for i in range(len(text) - 1, -1, -1):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == '}':
                brace_count += 1
            elif char == '{':
                brace_count -= 1
                if brace_count == 0:
                    last_complete_brace = i
                    break

    if last_complete_brace > 0:
        # 找到最后一个完整对象的位置
        text = text[:last_complete_brace + 1]

        # 找数组的结尾
        last_bracket = text.rfind(']')
        if last_bracket < 0 or last_bracket < last_complete_brace:
            text = text + ']'

    # 🔴 策略 2：如果还是解析失败，用正则提取完整对象（支持嵌套）
    try:
        import re
        # 更强的正则：匹配包含嵌套对象的完整 JSON 对象
        # 使用非贪婪匹配，支持 nested braces
        pattern = r'\{(?:[^{}]|(?R))*"algorithm"(?:[^{}]|(?R))*"confidence"(?:[^{}]|(?R))*\}'

        # Python 不支持 (?R)，使用替代方案：手动解析
        objects = []
        depth = 0
        start = -1

        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    obj_text = text[start:i+1]
                    # 验证是否包含必需字段
                    if '"algorithm"' in obj_text and '"confidence"' in obj_text:
                        objects.append(obj_text)
                    start = -1

        if objects:
            return '[' + ', '.join(objects) + ']'

    except Exception as e:
        pass

    # 🔴 策略 3：尝试提取所有看起来像 JSON 对象的部分
    try:
        import re
        # 简单的启发式：找到所有以 { 开头，包含 algorithm 和 confidence 的段落
        lines = text.split('\n')
        current_obj = []
        in_obj = False
        brace_depth = 0

        for line in lines:
            if '{' in line and not in_obj:
                in_obj = True
                current_obj = [line]
                brace_depth = line.count('{') - line.count('}')
            elif in_obj:
                current_obj.append(line)
                brace_depth += line.count('{') - line.count('}')

                if brace_depth == 0:
                    # 对象结束
                    obj_text = '\n'.join(current_obj)
                    if '"algorithm"' in obj_text and '"confidence"' in obj_text:
                        try:
                            # 尝试清理并解析
                            obj_text = obj_text.strip()
                            if obj_text.endswith(','):
                                obj_text = obj_text[:-1]
                            objects.append(obj_text)
                        except:
                            pass
                    in_obj = False
                    current_obj = []

        if objects:
            return '[' + ', '.join(objects) + ']'
    except:
        pass

    return text


def parse_recommendations(response_text: str, verbose: bool = False) -> List[Dict]:
    """解析 LLM 返回的推荐结果"""
    text = response_text.strip()

    # 移除 markdown 代码块标记
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(line for line in lines if not line.strip().startswith("```"))

    # 移除所有非 JSON 的前缀说明
    json_start = text.find("[")
    if json_start > 0:
        text = text[json_start:]

    # 尝试解析 JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            validated = _validate_recommendations(data)
            if validated:
                return validated
    except json.JSONDecodeError as e:
        if verbose:
            print(f"[LLM Agent] JSON 解析错误：{e}")
            print(f"[LLM Agent] 原始响应：{text[:200]}...")

    # 🔴 备用解析 1：尝试修复截断的 JSON
    try:
        fixed_text = _repair_truncated_json(text)
        if fixed_text != text:
            if verbose:
                print(f"[LLM Agent] 尝试修复截断的 JSON...")
            data = json.loads(fixed_text)
            if isinstance(data, list):
                validated = _validate_recommendations(data)
                if validated:
                    if verbose:
                        print(f"[LLM Agent] 修复成功，解析 {len(validated)} 个推荐")
                    return validated
    except json.JSONDecodeError:
        pass

    # 🔴 备用解析 2：正则提取完整对象（更强力的正则）
    try:
        import re
        # 匹配完整的 JSON 对象（支持嵌套）
        pattern = r'\{[^{}]*"algorithm"[^{}]*"confidence"\s*:\s*[\d.]+[^{}]*\}'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            validated = []
            for match in matches:
                try:
                    # 清理匹配结果
                    match = match.strip()
                    if not match.endswith('}'):
                        match = match + '}'
                    item = json.loads(match)
                    if isinstance(item, dict) and "algorithm" in item and "confidence" in item:
                        validated.append(item)
                except:
                    continue
            if validated:
                if verbose:
                    print(f"[LLM Agent] 通过正则提取 {len(validated)} 个推荐")
                return validated
    except Exception:
        pass

    if verbose:
        print(f"[LLM Agent] 警告：LLM 返回格式不正确，使用默认推荐")
    return []


def _validate_recommendations(data: list) -> List[Dict]:
    """验证推荐结果的格式"""
    validated = []

    for item in data:
        if not isinstance(item, dict):
            continue
        if "algorithm" not in item or "confidence" not in item:
            continue

        # 🔴 清理 algorithm 名称：移除括号、空格、特殊字符
        algo_name = item["algorithm"]
        # 移除括号及内容
        if '(' in algo_name:
            algo_name = algo_name.split('(')[0]
        # 移除多余空格
        algo_name = algo_name.strip().lower().replace(' ', '_').replace('-', '_')
        # 确保是有效的算法名格式（只保留字母、数字、下划线）
        import re
        algo_name = re.sub(r'[^a-z0-9_]', '', algo_name)

        # 🔴 确保算法名称以 _matching 结尾（LLM 已知道这个规范）
        if not algo_name.endswith('_matching'):
            algo_name = algo_name + '_matching'

        item["algorithm"] = algo_name

        # 🔴 修复 confidence 类型
        try:
            item["confidence"] = float(item["confidence"])
        except (ValueError, TypeError):
            item["confidence"] = 0.5  # 默认值

        # 确保 confidence 在 0-1 之间
        item["confidence"] = max(0.0, min(1.0, item["confidence"]))

        # 确保有 initial_params
        if "initial_params" not in item:
            item["initial_params"] = {}

        validated.append(item)

    return validated


def get_default_recommendations() -> List[Dict]:
    """默认推荐（当 LLM 失败时）"""
    return [
        {
            "algorithm": "feature_matching",
            "confidence": 0.8,
            "reason": "Default recommendation",
            "initial_params": {"detector": "SIFT", "matcher": "BF"}
        },
        {
            "algorithm": "template_matching",
            "confidence": 0.7,
            "reason": "Simple and fast",
            "initial_params": {"method": "TM_CCOEFF_NORMED"}
        }
    ]


def suggest_tuned_params(algo_name: str, history: List[Dict], verbose: bool = False) -> Dict:
    """
    使用 LLM 建议调优后的参数
    """
    history_str = "\n".join(
        f"尝试 {i+1}: params={h['params']} → confidence={h['confidence']:.3f}, found={h['found']}"
        for i, h in enumerate(history)
    )

    system_prompt = """你是计算机视觉算法参数调优专家。
根据之前的参数尝试和结果，建议改进的参数。

规则：
- 只建议算法实际使用的参数
- 努力提高置信度分数
- 如果所有尝试都失败了（found=false），建议完全不同的方法

仅返回一个 JSON 对象，格式如下：
{"param1": "value1", "param2": "value2"}

如果你无法建议改进，返回：{"stop": true}"""

    user_prompt = f"""算法：{algo_name}

之前的尝试：
{history_str}

为这个算法建议改进的参数。"""

    response = call_qwen_text(system_prompt, user_prompt)

    if response is None:
        return {}

    try:
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(line for line in lines if not line.startswith("```"))

        params = json.loads(text)

        if "stop" in params and params["stop"]:
            return {}

        return params
    except json.JSONDecodeError:
        return {}
