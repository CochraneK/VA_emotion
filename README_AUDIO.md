# VA Emotion Audio Analysis

This extension keeps the original project's flow:

`input stream -> emotion backend -> CSV -> HTML report`

The only change is replacing camera/video frames with microphone or audio-file windows.

## Install

```bat
pip install -r requirements.txt
pip install -r requirements_audio.txt
```

If you have an NVIDIA GPU, install the CUDA PyTorch build before installing the requirements.

## Real-time microphone

```bat
python src\audio_realtime.py --label patient_a
```

Useful options:

```bat
python src\audio_realtime.py --label patient_a --duration 120
python src\audio_realtime.py --segment-seconds 4 --hop-seconds 1
python src\audio_realtime.py --device 1
```

Stop with `Ctrl+C`. The script writes:

- `output\csv\audio_emotions_<label>_<timestamp>.csv`
- `output\reports\report_audio_emotions_<label>_<timestamp>.html`

## Audio file

```bat
python src\audio_file_to_csv.py --audio path\to\interview.wav --label patient_a
```

The default backend is Hugging Face `superb/wav2vec2-base-superb-er`, which predicts
`neutral`, `happy`, `sad`, and `angry`. The CSV keeps the same seven emotion columns
as the face pipeline, so unsupported emotions are filled with 0.

You can try another Hugging Face audio-classification model:

```bat
python src\audio_realtime.py --model your-org/your-audio-emotion-model
```

For clinical or research use, treat scores as behavioral signals rather than diagnosis.
Get consent before recording, and avoid sending sensitive audio to remote services.
