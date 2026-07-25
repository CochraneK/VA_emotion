# VA Emotion Analysis Toolkit

Video and audio emotion analysis toolkit for real-time webcam, local video, microphone, and audio-file workflows.

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
