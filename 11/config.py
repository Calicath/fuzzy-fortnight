import os

# Claude model
MODEL = "claude-opus-4-6"

# Offline / no-LLM mode
# 强制关闭离线模式 → 必须改成 False
OFFLINE = False

# LLM path: how many algorithms to recommend and run first
LLM_RECOMMEND_COUNT = 5
LLM_PRIORITY_RUN = 3   # run top N first before running the rest

# Auto-tuning: max iterations per algorithm
MAX_TUNE_ITERATIONS = 3

# Result confidence threshold (0-1) to consider a match "found"
MATCH_CONFIDENCE_THRESHOLD = 0.6

# API key from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ==============================================
# 🔴 关键修改：强制使用 qwen（通义千问）
# ==============================================
LLM_PROVIDER = "qwen"

# Qianwen (Qwen) / DashScope compatible mode settings (OpenAI-compatible API).
# ==============================================
# 🔴 关键修改：直接填入你的 API Key
# ==============================================
QWEN_API_KEY = "sk-e41f51d425434b68bdf8db2a26a83391"
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
# Use text-only model (no vision)
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-turbo-1101")

# Effective toggle for whether LLM path should be used.
if LLM_PROVIDER == "anthropic":
    USE_LLM = (not OFFLINE) and bool(ANTHROPIC_API_KEY)
elif LLM_PROVIDER == "qwen":
    USE_LLM = (not OFFLINE) and bool(QWEN_API_KEY)
else:
    USE_LLM = False