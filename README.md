<div id="readme-root">

<!-- ===== BILINGUAL SWITCH STYLES ===== -->
<style>
  /* Hide the radio inputs */
  .lang-switch {
    display: none;
  }

  /* Tab button base style */
  .lang-tab {
    display: inline-block;
    padding: 8px 28px;
    margin: 0 4px;
    font-size: 15px;
    font-weight: 600;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    border: 2px solid #0366d6;
    border-radius: 24px;
    cursor: pointer;
    color: #0366d6;
    background: #fff;
    transition: all 0.2s ease;
    user-select: none;
    text-decoration: none;
  }

  .lang-tab:hover {
    background: #f1f8ff;
  }

  /* Active English tab */
  #lang-en:checked ~ .tab-row label[for="lang-en"] {
    background: #0366d6;
    color: #fff;
  }

  /* Active Chinese tab */
  #lang-zh:checked ~ .tab-row label[for="lang-zh"] {
    background: #0366d6;
    color: #fff;
  }

  /* Hide both content areas by default */
  #en-content, #zh-content {
    display: none;
    text-align: left;
    margin-top: 24px;
  }

  /* Show English when English radio is checked */
  #lang-en:checked ~ #en-content {
    display: block;
  }

  /* Show Chinese when Chinese radio is checked */
  #lang-zh:checked ~ #zh-content {
    display: block;
  }

  /* Tab bar row */
  .tab-row {
    margin: 16px 0 8px 0;
  }

  /* Shared sections are always visible */
  .shared-section {
    text-align: left;
    margin-top: 24px;
  }

  /* Screenshot table */
  .screenshot-table {
    width: 100%;
    border-collapse: collapse;
  }
  .screenshot-table td {
    padding: 12px 16px;
    vertical-align: top;
    border-bottom: 1px solid #e1e4e8;
  }
  .screenshot-table td:first-child {
    width: 38%;
    font-weight: 600;
  }
  .screenshot-table td:last-child {
    width: 62%;
  }
  .screenshot-table img {
    max-width: 100%;
    border: 1px solid #d1d5da;
    border-radius: 6px;
  }
  .screenshot-note {
    font-size: 12px;
    color: #6a737d;
    margin-top: 4px;
  }
</style>

<h1 align="center">VA Emotion Analysis Toolkit</h1>

<!-- Hidden radio inputs for language switching -->
<input type="radio" name="lang" id="lang-en" class="lang-switch" checked>
<input type="radio" name="lang" id="lang-zh" class="lang-switch">

<!-- Tab buttons -->
<div class="tab-row" align="center">
  <label for="lang-en" class="lang-tab">🌐 English</label>
  <label for="lang-zh" class="lang-tab">🇨🇳 中文</label>
</div>

<!-- ===== ENGLISH CONTENT ===== -->
<div id="en-content">

A multi-backend toolkit for real-time webcam/video emotion detection, audio emotion analysis, and emotion synchrony research. Supports face and speech analysis pipelines and produces interactive HTML reports with charts and statistics.

### Features

- **Real-time webcam detection** — Live face detection with emotion intensity bars overlaid on the video feed
- **Video file analysis** — Process MP4/AVI/etc. with configurable frame-skipping for speed; export to CSV
- **Audio emotion analysis** — Analyze microphone input or audio files with a speech emotion model; export to CSV and HTML reports
- **Interactive HTML reports** — Pie charts, stacked area timelines, per-emotion bar charts, raw data tables (offline, via embedded Plotly.js)
- **Emotion synchrony analysis** — Compare two session CSVs with 6 statistical metrics (Pearson r, cross-correlation, mutual information, phase locking, Granger causality, running correlation)
- **Multi-backend support** — Pick the engine that fits your hardware (CPU-only or GPU-accelerated)

### Backends

| Backend | Dependency | Hardware | Speed | Accuracy | Notes |
|---|---|---|---|---|---|
| **HSEmotion** (default for webcam) | `hsemotion` + `torch` | GPU (CUDA) recommended, CPU fallback | ~20 ms/frame (GPU) | High | EfficientNet-B0 on AffectNet, pure ML model |
| **MediaPipe** | `mediapipe` | CPU | Fast (~5 ms/frame) | Medium | Heuristic-based from Face Mesh landmarks |
| **InsightFace** | `insightface` | CPU | Medium | Medium | BuffaloS detector + geometry heuristic |
| **DeepFace** (optional) | `deepface` (+ TensorFlow) | GPU/CPU | Slow | High | Multiple model choices (VGG-Face, ArcFace, etc.) |

### Installation (pip)

**Prerequisites:** Python 3.10+ (recommended 3.12)

```bash
# 1. Create a virtual environment (venv or conda, your choice)
conda create -n face-emotion python=3.12 -y
conda activate face-emotion

# 2. Install PyTorch (CUDA version recommended if you have an NVIDIA GPU)
#    NVIDIA GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#    CPU only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install all dependencies
pip install -r requirements.txt

# 4. (Optional) DeepFace — requires TensorFlow
pip install deepface
```

**requirements.txt contents:**

```
opencv-python>=4.8
numpy>=1.24
torch>=2.0
hsemotion>=0.1
mediapipe>=0.10
insightface>=0.7
scipy>=1.10
```

> **Note:** There is no Docker support. This project is designed to run directly on your machine with pip-installed Python packages.

### Usage

#### 1. Real-time webcam

```bash
python src\webcam.py
# Session label for output file naming:
python src\webcam.py --label alice

# Controls: q = quit and save, s = screenshot
```

Generates: `output\csv\emotions_<label>_<timestamp>.csv` + auto-generated HTML report in `output\reports\`.

#### 2. Video file analysis

```bash
# Quick analysis with default (HSEmotion) backend
python src\video_to_csv.py --video path\to\video.mp4 --label alice --skip 10

# Unified runner — pick any backend
python src\run_all.py --video path\to\video.mp4 --backend mediapipe --interval 15
python src\run_all.py --video path\to\video.mp4 --backend insightface
python src\run_all.py --video path\to\video.mp4 --backend deepface
python src\run_all.py --video path\to\video.mp4 --backend all     # benchmark all backends

# Draw emotion labels on output video
python src\run_all.py --video path\to\video.mp4 --backend mediapipe --draw
```

- `--interval` / `--skip`: Analyze every N frames (default: 15). Smaller = more accurate, slower.

#### 3. HTML reports

Generated automatically after each webcam or video session. To regenerate manually:

```bash
python tools\emotion_report.py output\csv\emotions_alice_20260611_120000.csv
# Auto-find latest report for a label:
python tools\emotion_report.py --label alice
```

#### 4. Emotion synchrony analysis

Compare two session CSVs (e.g., two people in the same scene):

```bash
python tools\emotion_sync.py output\csv\emotions_alice_20260611_120000.csv output\csv\emotions_bob_20260611_120000.csv
```

**6 Metrics:**

| Metric | Range | Meaning |
|---|---|---|
| Pearson r | -1 ~ +1 | Linear correlation (+1 = perfect sync) |
| Cross-correlation lag | seconds | Who leads whom (negative = B leads, positive = A leads) |
| Mutual Information | 0 ~ 1 | Nonlinear shared information |
| Phase Locking Value | 0 ~ 1 | Are emotion oscillations in sync? |
| Granger Causality | F-stat, * = p<0.05 | Directional: does A's past predict B's future? |
| Running Correlation | -1 ~ +1 | How synchrony evolves over time (sliding window) |

### Project Structure

```
VA_emotion/
├── src/                          # Core source code
│   ├── webcam.py                 # Real-time webcam detection
│   ├── video_to_csv.py           # Video file -> CSV analysis
│   ├── run_all.py                # Unified video runner (multi-backend)
│   ├── compare.py                # Side-by-side backend comparison
│   ├── backend_hsemotion.py      # HSEmotion (EfficientNet-B0, GPU)
│   ├── backend_mediapipe.py      # MediaPipe Face Mesh (CPU)
│   ├── backend_insightface.py    # InsightFace BuffaloS (CPU)
│   ├── backend_deepface.py       # DeepFace (optional, TensorFlow)
│   └── face_detection_yunet_2023mar.onnx  # YuNet detector (auto-downloaded)
├── tools/                        # Utility scripts
│   ├── emotion_report.py         # HTML report generator
│   ├── emotion_sync.py           # Emotion synchrony analysis
│   └── setup_env.bat             # Conda setup helper
├── models\buffalo_s/             # InsightFace pre-trained models (included)
├── output/                       # Output directory
│   ├── csv/                      # Session CSV data
│   ├── reports/                  # HTML emotion reports
│   ├── sync/                     # Synchrony analysis reports
│   └── plotly-2.32.0.min.js      # Offline Plotly.js bundle
├── requirements.txt              # Python dependencies
├── webcam.bat                    # Quick-start webcam
├── video_analyze.bat             # Quick-start video analysis
└── README.md
```

### Troubleshooting

**Camera won't open:** Make sure no other app is using the camera. The code uses `cv2.CAP_DSHOW` backend by default.

**GPU out of memory:** Reduce `CAM_W` in `webcam.py` (default 480), or increase `SKIP` to analyze fewer frames.

**DeepFace not available:** DeepFace requires TensorFlow. Install separately: `pip install deepface`

**YuNet model download fails:** The tool auto-downloads on first run. If it fails, manually download to `src/face_detection_yunet_2023mar.onnx`:
<https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx>

</div>

<!-- ===== CHINESE CONTENT ===== -->
<div id="zh-content">

多后端人脸情绪分析工具集，支持**实时摄像头检测**、**视频文件分析**以及**双人情绪同步性研究**。提供四种后端引擎，自动生成交互式 HTML 报告（含饼图、时间线、柱状图、原始数据表）。

### 功能

- **实时摄像头检测** — 在视频画面上实时叠加人脸检测和情绪强度条
- **视频文件分析** — 支持 MP4/AVI 等格式，可跳帧加速处理，导出 CSV 数据
- **交互式 HTML 报告** — 饼图、堆叠时间线、情绪柱状图、原始数据表（离线可用，内嵌 Plotly.js）
- **情绪同步性分析** — 对两个会话 CSV 进行 6 项统计指标对比（双人同框场景）
- **多后端支持** — 根据硬件条件自由选择（纯 CPU 或 GPU 加速）

### 后端引擎

| 后端引擎 | 依赖 | 硬件要求 | 速度 | 精度 | 说明 |
|---|---|---|---|---|---|
| **HSEmotion**（摄像头默认） | `hsemotion` + `torch` | GPU (CUDA) 推荐，CPU 可降级 | ~20 ms/帧（GPU） | 高 | AffectNet 训练的 EfficientNet-B0，纯 ML 模型 |
| **MediaPipe** | `mediapipe` | CPU | 快 (~5 ms/帧) | 中 | 基于 Face Mesh 关键点的启发式规则 |
| **InsightFace** | `insightface` | CPU | 中 | 中 | BuffaloS 人脸检测 + 几何特征启发式 |
| **DeepFace**（可选） | `deepface` + TensorFlow | GPU/CPU | 慢 | 高 | 支持多种模型（VGG-Face、ArcFace 等） |

### 安装（pip）

**前提：** Python 3.10+（推荐 3.12）

```bash
# 1. 创建虚拟环境（conda 或 venv 均可）
conda create -n face-emotion python=3.12 -y
conda activate face-emotion

# 2. 安装 PyTorch（有 NVIDIA 显卡建议装 CUDA 版）
#    NVIDIA 显卡：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#    仅 CPU：
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. 安装所有依赖
pip install -r requirements.txt

# 4. （可选）DeepFace — 需要 TensorFlow
pip install deepface
```

**requirements.txt 内容：**

```
opencv-python>=4.8
numpy>=1.24
torch>=2.0
hsemotion>=0.1
mediapipe>=0.10
insightface>=0.7
scipy>=1.10
```

> **注意：** 本项目不提供 Docker 镜像，需通过 pip 直接在本地机器上安装 Python 依赖。

### 使用方法

#### 1. 实时摄像头检测

```bash
python src\webcam.py
# 添加会话标签，便于输出文件区分：
python src\webcam.py --label alice

# 快捷键: q = 退出并保存，s = 截图
```

输出：`output\csv\emotions_<标签>_<时间戳>.csv` + 自动在 `output\reports\` 生成 HTML 报告。

#### 2. 视频文件分析

```bash
# 使用默认后端（HSEmotion）快速分析
python src\video_to_csv.py --video 视频路径.mp4 --label alice --skip 10

# 统一分析器 — 选择任意后端
python src\run_all.py --video 视频路径.mp4 --backend mediapipe --interval 15
python src\run_all.py --video 视频路径.mp4 --backend insightface
python src\run_all.py --video 视频路径.mp4 --backend deepface
python src\run_all.py --video 视频路径.mp4 --backend all     # 对比所有后端

# 在输出视频上绘制情绪标签
python src\run_all.py --video 视频路径.mp4 --backend mediapipe --draw
```

- `--interval` / `--skip`：每 N 帧分析一次（默认 15）。数值越小越精确但也越慢。

#### 3. HTML 报告

摄像头或视频分析后自动生成。手动重新生成：

```bash
python tools\emotion_report.py output\csv\emotions_alice_20260611_120000.csv
# 自动查找某个标签的最新报告：
python tools\emotion_report.py --label alice
```

#### 4. 情绪同步性分析

对比两个会话 CSV（例如同一场景下的两个人）：

```bash
python tools\emotion_sync.py output\csv\emotions_alice_20260611_120000.csv output\csv\emotions_bob_20260611_120000.csv
```

**6 项统计指标：**

| 指标 | 范围 | 含义 |
|---|---|---|
| 皮尔逊相关系数 (Pearson r) | -1 ~ +1 | 线性相关性，+1 为完美同步 |
| 互相关延迟 (Cross-correlation lag) | 秒 | 谁领先谁（负值=B领先A，正值=A领先B） |
| 互信息 (Mutual Information) | 0 ~ 1 | 非线性共享信息量 |
| 锁相值 (Phase Locking Value) | 0 ~ 1 | 情绪振荡是否同步 |
| 格兰杰因果 (Granger Causality) | F统计量，* 表示 p<0.05 | 方向性：A 的历史能否预测 B 的未来？ |
| 滑动窗口相关 (Running Correlation) | -1 ~ +1 | 同步性随时间的变化（滑动窗口） |

### 项目结构

```
VA_emotion/
├── src/                          # 核心源代码
│   ├── webcam.py                 # 实时摄像头情绪检测
│   ├── video_to_csv.py           # 视频转 CSV 分析
│   ├── run_all.py                # 统一视频分析器（支持多后端）
│   ├── compare.py                # 多后端对比工具
│   ├── backend_hsemotion.py      # HSEmotion (EfficientNet-B0, GPU)
│   ├── backend_mediapipe.py      # MediaPipe Face Mesh (CPU)
│   ├── backend_insightface.py    # InsightFace BuffaloS (CPU)
│   ├── backend_deepface.py       # DeepFace (可选, 需 TensorFlow)
│   └── face_detection_yunet_2023mar.onnx  # YuNet 检测模型（首次运行自动下载）
├── tools/                        # 辅助工具
│   ├── emotion_report.py         # HTML 报告生成器
│   ├── emotion_sync.py           # 情绪同步性分析器
│   └── setup_env.bat             # conda 配置辅助
├── models\buffalo_s/             # InsightFace 预训练模型（已包含）
├── output/                       # 输出目录
│   ├── csv/                      # 会话 CSV 数据
│   ├── reports/                  # HTML 情绪报告
│   ├── sync/                     # 同步性分析报告
│   └── plotly-2.32.0.min.js      # 离线 Plotly.js 文件
├── requirements.txt              # Python 依赖清单
├── webcam.bat                    # 快速启动摄像头
├── video_analyze.bat             # 快速启动视频分析
└── README.md
```

### 常见问题

**摄像头无法打开：** 确保没有其他程序占用摄像头。代码默认使用 `cv2.CAP_DSHOW` 后端。

**GPU 显存不足：** 减小 `webcam.py` 中的 `CAM_W`（默认 480），或增大 `SKIP` 减少分析帧数。

**DeepFace 不可用：** DeepFace 需要 TensorFlow，需单独安装：`pip install deepface`

**YuNet 模型下载失败：** 工具会在首次运行时自动下载。如果失败，手动下载到 `src/face_detection_yunet_2023mar.onnx`：
<https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx>

</div>

<!-- ===== HR ===== -->
<hr>

<!-- ===== SHARED: SCREENSHOTS / 运行截图 ===== -->
<div class="shared-section">

## 📸 Screenshots / 运行截图

<table class="screenshot-table">
  <tr>
    <td><code>webcam_single.bat</code> 运行效果<br><small>Real-time webcam with emotion intensity bars</small></td>
    <td>
      <img src="screenshots/placeholder.png" alt="webcam_single screenshot" width="400">
      <div class="screenshot-note">(待替换为实际运行截图 / Replace with actual screenshot)</div>
    </td>
  </tr>
  <tr>
    <td><code>webcam_multi.bat</code> 运行效果<br><small>Multi-face simultaneous detection</small></td>
    <td>
      <img src="screenshots/placeholder.png" alt="webcam_multi screenshot" width="400">
      <div class="screenshot-note">(待替换为实际运行截图 / Replace with actual screenshot)</div>
    </td>
  </tr>
  <tr>
    <td><code>video_analyze.bat</code> 控制台交互<br><small>Drag-and-drop video path interface</small></td>
    <td>
      <img src="screenshots/placeholder.png" alt="video_analyze screenshot" width="400">
      <div class="screenshot-note">(待替换为实际运行截图 / Replace with actual screenshot)</div>
    </td>
  </tr>
  <tr>
    <td><code>video_sync.bat</code> 控制台交互<br><small>CSV file list selection menu</small></td>
    <td>
      <img src="screenshots/placeholder.png" alt="video_sync screenshot" width="400">
      <div class="screenshot-note">(待替换为实际运行截图 / Replace with actual screenshot)</div>
    </td>
  </tr>
  <tr>
    <td>HTML 情绪报告 - 概览页<br><small>Report overview: pie chart + summary statistics</small></td>
    <td>
      <img src="screenshots/placeholder.png" alt="emotion report overview" width="400">
      <div class="screenshot-note">(待替换为实际运行截图 / Replace with actual screenshot)</div>
    </td>
  </tr>
  <tr>
    <td>HTML 情绪报告 - 时间线<br><small>Report timeline: stacked area chart</small></td>
    <td>
      <img src="screenshots/placeholder.png" alt="emotion report timeline" width="400">
      <div class="screenshot-note">(待替换为实际运行截图 / Replace with actual screenshot)</div>
    </td>
  </tr>
  <tr>
    <td>HTML 情绪报告 - 柱状图<br><small>Report bar chart: per-emotion distribution</small></td>
    <td>
      <img src="screenshots/placeholder.png" alt="emotion report bar chart" width="400">
      <div class="screenshot-note">(待替换为实际运行截图 / Replace with actual screenshot)</div>
    </td>
  </tr>
  <tr>
    <td>同步性分析报告<br><small>Synchrony analysis: 6 statistical metrics output</small></td>
    <td>
      <img src="screenshots/placeholder.png" alt="synchrony analysis report" width="400">
      <div class="screenshot-note">(待替换为实际运行截图 / Replace with actual screenshot)</div>
    </td>
  </tr>
</table>

</div>

<!-- ===== HR ===== -->
<hr>

<!-- ===== SHARED: REFERENCES / 参考文献 ===== -->
<div class="shared-section">

## 📚 References / 参考文献

- **HSEmotion** — Savchenko, CVPR 2022. [arXiv:2108.01588](https://arxiv.org/abs/2108.01588)
- **MediaPipe** — Google. [mediapipe.dev](https://mediapipe.dev)
- **InsightFace** — InsightFace Team. [github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)
- **AffectNet** — Mollahosseini et al., 2017. 450k+ labeled face images.

</div>

</div>
<!-- END #readme-root -->
