# VA Emotion Analysis Toolkit

Video and audio emotion analysis toolkit for real-time webcam, local video, microphone, and audio-file workflows.

<p align="center">
  <a href="#english"><img src="https://img.shields.io/badge/English-VA_emotion-blue?style=for-the-badge" alt="English"></a>
  <a href="#chinese"><img src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-VA_emotion-green?style=for-the-badge" alt="中文"></a>
</p>

<a name="english"></a>
## English

## Features

- Real-time webcam emotion detection
- Local video file analysis
- Real-time microphone speech emotion analysis
- Audio file emotion analysis
- HTML reports for every session
- Emotion synchrony analysis between two CSV sessions

## Install

```bash
conda create -n face-emotion python=3.12 -y
conda activate face-emotion

pip install -r requirements.txt
pip install -r requirements_audio.txt
```

If you use NVIDIA GPU acceleration, install the CUDA build of PyTorch first.

## Quick Start

### Video

```bash
python src\webcam.py --label alice
python src\video_to_csv.py --video path\to\video.mp4 --label alice --skip 10
python src\run_all.py --video path\to\video.mp4 --backend mediapipe --interval 15
```

### Audio

```bash
python src\audio_realtime.py --label patient_a
python src\audio_file_to_csv.py --audio path\to\audio.wav --label patient_a
```

## BAT shortcuts

- `webcam_single.bat`
- `webcam_multi.bat`
- `video_analyze.bat`
- `video_sync.bat`
- `audio_realtime.bat`
- `audio_analyze.bat`
- `audio_sync.bat`
- `window_emotion.bat`

## Outputs

- CSV: `output\csv\`
- Session reports: `output\reports\`
- Synchrony reports: `output\sync\`

## Project Structure

```text
VA_emotion/
+-- src/
|   +-- webcam.py
|   +-- webcam_multi.py
|   +-- video_to_csv.py
|   +-- run_all.py
|   +-- window_emotion.py
|   +-- audio_realtime.py
|   +-- audio_file_to_csv.py
|   +-- backend_hsemotion.py
|   +-- backend_mediapipe.py
|   +-- backend_insightface.py
|   +-- backend_deepface.py
|   +-- backend_audio_transformers.py
+-- tools/
|   +-- emotion_report.py
|   +-- emotion_sync.py
+-- output/
+-- docs/
+-- requirements.txt
+-- requirements_audio.txt
```

## Notes

- Video and audio are supported as separate analysis pipelines.
- `video_sync.bat` compares two video-session CSV files.
- `audio_sync.bat` compares two audio-session CSV files.
- The default audio backend is a Hugging Face speech emotion model.
- The original face pipeline still works and keeps its CSV/report format.

## Links

- GitHub: [CochraneK/VA_emotion](https://github.com/CochraneK/VA_emotion)

---

<a name="chinese"></a>
## 中文

这是一个用于视频和音频情绪分析的工具集，支持摄像头、本地视频、麦克风实时语音、音频文件分析，并可生成 HTML 报告和同步性分析结果。

### 功能

- 实时摄像头情绪检测
- 本地视频文件分析
- 麦克风实时语音情绪分析
- 音频文件情绪分析
- 每次会话自动生成 HTML 报告
- 两个 CSV 会话之间的同步性分析

### 安装

```bash
conda create -n face-emotion python=3.12 -y
conda activate face-emotion

pip install -r requirements.txt
pip install -r requirements_audio.txt
```

如果使用 NVIDIA 显卡，建议先安装 CUDA 版 PyTorch。

### 快速开始

#### 视频

```bash
python src\webcam.py --label alice
python src\video_to_csv.py --video path\to\video.mp4 --label alice --skip 10
python src\run_all.py --video path\to\video.mp4 --backend mediapipe --interval 15
```

#### 音频

```bash
python src\audio_realtime.py --label patient_a
python src\audio_file_to_csv.py --audio path\to\audio.wav --label patient_a
```

### BAT 快捷入口

- `webcam_single.bat`
- `webcam_multi.bat`
- `video_analyze.bat`
- `video_sync.bat`
- `audio_realtime.bat`
- `audio_analyze.bat`
- `audio_sync.bat`
- `window_emotion.bat`

### 输出目录

- CSV：`output\csv\`
- 会话报告：`output\reports\`
- 同步性报告：`output\sync\`

### 说明

- 视频和音频是分开的分析流程。
- `video_sync.bat` 用于对比两个视频会话 CSV。
- `audio_sync.bat` 用于对比两个音频会话 CSV。
- 默认音频后端是 Hugging Face 的语音情绪模型。
- 原始人脸情绪流程仍然保留。
