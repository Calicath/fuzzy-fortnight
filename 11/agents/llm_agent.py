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


def run_llm_recommendation(template: cv2.Mat, scene: cv2.Mat, verbose: bool = False) -> List[Dict]:
    """
    使用 LLM 推荐算法
    """

    # 1. 提取图像特征描述（文本）
    template_desc = extract_image_features(template)
    scene_desc = extract_image_features(scene)

    # 2. 构建提示词（纯文本）
    system_prompt = """You are an expert algorithm engineer.
Analyze the image descriptions and recommend image matching / comparison algorithms.

IMPORTANT RULES:
- Do NOT recommend deep learning or neural network based methods (no SuperPoint, SuperGlue, LoFTR, CNN-based, etc.)
- Recommend ANY suitable non-deep-learning algorithm: traditional CV, signal processing, statistical methods, etc.
- The algorithm must have a pip-installable Python implementation

Example categories you can recommend (but NOT limited to):
- Feature-based: SIFT, SURF, ORB, AKAZE, FAST, BRISK, etc.
- Template matching: sliding window correlation, FFT-based matching
- Frequency domain: phase correlation, Fourier transform matching
- Image similarity: SSIM, histogram comparison, Earth Mover's Distance
- Statistical: mutual information, normalized cross correlation, Hausdorff distance
- Keypoint detection: Harris, Shi-Tomasi, etc.
- Any other non-DL algorithm available via pip

For each recommended algorithm, provide:
1. algorithm name (snake_case, e.g., "sift_matching", "histogram_comparison")
2. confidence score (0-1)
3. brief reason
4. initial parameters

Respond ONLY with a JSON array like:
[
  {
    "algorithm": "feature_matching",
    "confidence": 0.95,
    "reason": "Template contains rich texture features",
    "initial_params": {"detector": "SIFT", "matcher": "BF"}
  },
  ...
]

Recommend 3-5 algorithms sorted by confidence."""

    user_prompt = f"""Template Image Description:
{template_desc}

Scene Image Description:
{scene_desc}

What algorithms would you recommend for matching these images?"""

    if verbose:
        print("[LLM Agent] 正在分析图像特征...")

    # 调用 qwen 文本模型
    if verbose:
        print("[LLM Agent] 调用 qwen 文本模型...")

    response_text = call_qwen_text(system_prompt, user_prompt)

    if response_text is None:
        if verbose:
            print("[LLM Agent] LLM 调用失败，使用默认推荐")
        return get_default_recommendations()

    # 解析 JSON 结果
    recommendations = parse_recommendations(response_text)

    if verbose and recommendations:
        print(f"[LLM Agent] 成功推荐 {len(recommendations)} 个算法")

    return recommendations or get_default_recommendations()


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
        "temperature": 0.1,  # 降低温度，让输出更稳定
        "max_tokens": 2000,  # 🔴 确保返回完整的 JSON
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    # 应用速率限制
    _rate_limit()
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            # 处理 429 限流错误
            if response.status_code == 429:
                wait_time = 2 ** (attempt + 1)  # 指数退避: 2s, 4s, 8s
                print(f"[LLM Agent] 遇到限流（429），等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
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


def parse_recommendations(response_text: str) -> List[Dict]:
    """解析 LLM 返回的推荐结果"""
    text = response_text.strip()
    
    # 移除 markdown 代码块标记
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(line for line in lines if not line.startswith("```"))
    
    # 移除所有非 JSON 的前缀说明
    json_start = text.find("[")
    if json_start > 0:
        text = text[json_start:]
    
    # 尝试解析 JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            # 验证每个推荐的格式
            validated = []
            for item in data:
                if isinstance(item, dict) and "algorithm" in item and "confidence" in item:
                    validated.append(item)
            if validated:
                return validated
    except json.JSONDecodeError as e:
        print(f"[LLM Agent] JSON 解析错误：{e}")
        print(f"[LLM Agent] 原始响应：{text[:200]}...")
    
    # 🔴 备用解析 1：尝试修复不完整的 JSON
    try:
        # 检查 JSON 是否被截断（以 ... 结尾）
        if text.rstrip().endswith('...') or text.rstrip().endswith('"'):
            print(f"[LLM Agent] 检测到 JSON 被截断，尝试修复...")
            # 尝试找到最后一个完整的对象
            last_brace = text.rfind('}')
            last_bracket = text.rfind(']')
            if last_brace > 0 and (last_bracket < 0 or last_brace < last_bracket):
                # 对象在数组内，截断在对象中
                text = text[:last_brace+1] + ']}'  # 尝试补全
            elif last_bracket > 0:
                # 截断在数组级别
                text = text[:last_bracket+1]
            
            data = json.loads(text)
            if isinstance(data, list):
                validated = []
                for item in data:
                    if isinstance(item, dict) and "algorithm" in item and "confidence" in item:
                        validated.append(item)
                if validated:
                    print(f"[LLM Agent] 成功修复并解析 {len(validated)} 个推荐")
                    return validated
    except json.JSONDecodeError:
        pass
    
    # 备用解析 2：尝试提取 JSON 数组
    try:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            json_text = text[start:end]
            data = json.loads(json_text)
            if isinstance(data, list):
                validated = []
                for item in data:
                    if isinstance(item, dict) and "algorithm" in item and "confidence" in item:
                        validated.append(item)
                if validated:
                    return validated
    except json.JSONDecodeError:
        pass
    
    # 🔴 备用解析 3：尝试逐行解析（处理流式输出）
    try:
        # 尝试找到所有完整的 JSON 对象
        import re
        pattern = r'\{[^}]*"algorithm"[^}]*"confidence"[^}]*\}'
        matches = re.findall(pattern, text)
        if matches:
            validated = []
            for match in matches:
                try:
                    item = json.loads(match)
                    if "algorithm" in item and "confidence" in item:
                        validated.append(item)
                except:
                    continue
            if validated:
                print(f"[LLM Agent] 通过正则提取 {len(validated)} 个推荐")
                return validated
    except Exception:
        pass
    
    print(f"[LLM Agent] 警告：LLM 返回格式不正确，使用默认推荐")
    return []


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

    system_prompt = """You are an algorithm tuning expert.
Based on previous parameter attempts and results, suggest improved parameters.

Rules:
- Only suggest parameters that the algorithm actually uses
- Try to improve confidence score
- If all attempts failed (found=false), suggest a completely different approach

Respond ONLY with a JSON object like:
{"param1": "value1", "param2": "value2"}

If you cannot suggest improvements, respond with: {"stop": true}"""

    user_prompt = f"""Algorithm: {algo_name}

Previous attempts:
{history_str}

Suggest improved parameters for this algorithm."""

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
