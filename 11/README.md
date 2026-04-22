# 🖼️ Image Matcher Agent - 智能图像匹配 Agent

基于 LLM 和 OpenCV 的智能图像匹配系统，具备**自动选择算法、自动下载代码、自动调参**的全自动化能力。

## ✨ 核心特性

### 🤖 全自动化 Agent
- **自动选择算法**：根据图像特征和场景，智能推荐最适合的匹配算法
- **自动下载代码**：通过 Gitee/GitHub 搜索算法实现，或 LLM 自动生成调用代码
- **自动安装依赖**：检测并安装缺失的 Python 包（如 opencv-python）
- **自动调参优化**：通过二分搜索自动寻找最优参数组合，提升匹配准确率
- **智能缓存**：已下载的算法自动缓存，避免重复下载

### 📊 多种匹配算法
系统支持多种 OpenCV 图像匹配算法：
- **SIFT**：尺度不变特征变换（高精度）
- **ORB**：Oriented FAST and Rotated BRIEF（快速）
- **AKAZE**：加速 KAZE（中等速度）
- **BRISK**：二进制鲁棒不变尺度关键点
- **SURF**：加速鲁棒特征（需要 opencv-contrib-python）
- **模板匹配**：cv2.matchTemplate
- **直方图比较**：cv2.compareHist

### 🌐 Web 界面
- 现代化的 Web UI，支持拖拽上传
- 实时显示匹配结果和置信度
- 匹配可视化（特征点连线）
- 详细的匹配报告

## 📁 项目结构

```
image-matcher-agent/
├── agents/                          # Agent 核心模块
│   ├── __init__.py                  # 包初始化
│   ├── orchestrator.py              # 🎯 调度器（核心）- 协调所有 Agent 工作
│   ├── llm_agent.py                 # 🧠 LLM Agent - 与大语言模型交互
│   ├──               
│   ├── match_result.py              # 匹配结果数据类
│   └── algorithm_downloader.py      # 📥 算法下载器 - 自动获取算法代码
│
├── tools/                           # 工具模块
│   ├── __init__.py                  # 包初始化
│   ├── image_loader.py              # 图像加载工具
│   ├── image_selector.py            # 图像选择/预览工具
│   ├── param_tuner.py               # 🔧 参数调优器 - 自动二分搜索调参
│   ├── visualizer.py                # 匹配可视化工具
│   └── reporter.py                  # 匹配报告生成器
│
├── web/                             # Web 界面
│   ├── app.py                       # Flask Web 应用
│   └── templates/
│       └── index.html               # Web UI 模板
│
├── config.py                        # 全局配置（LLM API 地址等）
├── main.py                          # CLI 入口
├── requirements.txt                 # Python 依赖
└── 启动 Web 界面.bat                 # Windows 快速启动脚本
```

## 🏗️ 技术架构

### 1. Agent 架构设计

系统采用**多 Agent 协同架构**，各 Agent 各司其职：

```
用户请求
  ↓
[调度器 Orchestrator] ← 核心协调者
  ↓
  ├── [LLM Agent] ──── 与大语言模型对话
  │     ├── 推荐算法（基于场景分析）
  │     └── 生成算法调用代码
  │
  ├── [算法下载器] ─── 自动获取算法实现
  │     ├── 检查缓存
  │     ├── pip 安装依赖
  │     ├── LLM 生成代码
  │     └── Gitee 搜索下载
  │
  └── [参数调优器] ── 自动优化参数
        └── 二分搜索最优参数
```

### 2. 自动选择算法流程

**实现位置**：`agents/llm_path.py` + `agents/orchestrator.py`

```python
# LLM 根据以下因素推荐算法：
1. 图像类型（自然风景、建筑物、文字等）
2. 场景特征（光照变化、尺度变化、旋转等）
3. 性能要求（速度 vs 精度）
4. 历史成功率统计
```

**流程**：
1. 用户描述图像场景
2. 调度器发送场景描述给 LLM Agent
3. LLM 分析场景特征，推荐 3-5 个算法并给出置信度
4. 按置信度排序，依次尝试

### 3. 自动下载算法实现

**实现位置**：`agents/algorithm_downloader.py`

**获取策略**（按优先级）：

```
1. 检查缓存 → 已有直接加载
   ↓
2. 有 pip 库？ → pip install + LLM 生成调用代码
   ↓
3. Gitee 搜索 → 搜索 Python 实现并下载
   ↓
4. 全部失败 → 返回 None
```

**技术细节**：
- **缓存机制**：下载的代码保存到 `downloaded_algorithms/`，通过文件内容哈希命名
- **LLM 代码生成**：让 LLM 生成完整的调用代码，包含：
  ```python
  def run(template_image, scene_image):
      # 自动处理灰度化、特征提取、匹配、置信度计算
      ...
  
  def default_params():
      return {...}  # 定义可调参数
  ```
- **依赖自动安装**：检测 `pip_list()`，缺失则自动 `pip install`
- **代码验证**：下载后编译验证语法正确性

### 4. 自动调参优化

**实现位置**：`tools/param_tuner.py`

**核心算法**：**坐标二分搜索（Coordinate Binary Search）**

```python
# 调参流程：
for param in params:
    # 1. 二分搜索最优值
    while 区间 > 阈值:
        测试两个候选值
        保留置信度更高的
    
    # 2. 验证参数有效性
    if 参数不影响结果:
        跳过该参数
    
    # 3. 更新最优结果
    best_params[param] = best_value
```

**调参策略**：
- **数值型参数**（如 threshold）：二分搜索 [0.0, 1.0]
- **分类型参数**（如 detector 类型）：遍历所有候选值
- **动态限制**：最多调优 5 个关键参数，避免参数过多
- **提前终止**：如果参数不影响结果，自动跳过

### 5. 完整工作流

```
1. 用户上传图像 → 系统接收模板图和场景图
   ↓
2. LLM 分析场景 → 推荐算法列表及置信度
   ↓
3. 调度器依次尝试每个算法：
   ├─ 获取算法实现（缓存/LLM/下载）
   ├─ 执行算法 → 获得初始置信度
   └─ 如果置信度 < 0.5，尝试调优
   ↓
4. 参数调优（如果需要）：
   ├─ 分析可调参数
   ├─ 二分搜索最优值
   └─ 更新参数配置
   ↓
5. 选择最佳结果 → 生成可视化和报告
   ↓
6. 返回给用户
```

## 🔧 使用的技术

| 技术 | 用途 |
|------|------|
| **OpenCV** | 图像处理、特征提取、匹配算法 |
| **LLM API** | 算法推荐、代码生成（兼容 OpenAI API） |
| **Flask** | Web 框架 |
| **NumPy** | 数值计算 |
| **Requests** | HTTP 请求（下载算法、调用 LLM） |
| **Subprocess** | pip 安装依赖 |
| **Importlib** | 动态加载算法模块 |
| **Hashlib** | 缓存文件命名（SHA256） |



## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置 LLM API

编辑 `config.py`，设置 LLM API 地址：

```python
LLM_CONFIG = {
    "api_base": "http://your-api-url/v1",
    "api_key": "your-api-key",
    "model": "your-model-name"
}
```

### 启动 Web 界面

```bash
# 方式 1：双击启动
启动 Web 界面.bat

# 方式 2：命令行
python web/app.py
```

### CLI 模式

```bash
python main.py
```

## 📝 依赖说明

运行前需要先安装依赖：
```bash
pip install -r requirements.txt
```

## 🤝 项目亮点

1. **真正的全自动化**：从算法选择到代码获取到调参，全程无需人工干预
2. **智能容错**：缓存 → LLM → 网络搜索，多层后备策略
3. **自适应调参**：自动识别参数类型，采用最优搜索策略
4. **可扩展架构**：Agent 模块化设计，易于添加新算法
5. **开箱即用**：配置好 LLM API 后即可使用，无需手动安装算法库
