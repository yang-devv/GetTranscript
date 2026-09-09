# GetTranscript

A local-first audio transcription toolkit. Download, transcribe, and polish — from YouTube video to publication-ready transcript in three commands.

## Pipeline

1. **YouTube Auto-Download** (`youtube_to_wav.py`) — Pass a video or playlist URL. Uses yt-dlp to extract the highest quality WAV audio automatically.
2. **Local Whisper Transcription** (`transcribe_wav.py` / `transcribe.py`) — Runs OpenAI's Whisper large-v3 locally via faster-whisper. Supports GPU batch inference, checkpoint resume, and float16/int8 precision modes.
3. **Intelligent Punctuation** (`batch_punctuate.py`) — LLM post-processor that adds punctuation, paragraph breaks, and natural formatting to produce publication-ready transcripts.

## Setup

### 1. Create Conda Environment
```bash
conda create -n whisper_env python=3.10 -y
conda activate whisper_env
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg
- **Windows**: `winget install FFmpeg` or `scoop install ffmpeg`
- **Mac**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

## Hardware Requirements

- **Recommended**: NVIDIA GPU with 8GB+ VRAM (RTX 3070/4060 or above) for float16 inference
- **Lower VRAM (4-6GB)**: Change `compute_type` to `int8_float16` or `int8`
- **No GPU**: Falls back to CPU mode (much slower)

If you encounter missing `cudnn` DLL errors on Windows, see [this guide](https://joshuachini.com/2024/02/04/how-to-setup-whisper-from-openai/).

## API Key

The punctuation script (`batch_punctuate.py`) uses the DeepSeek API. Get a key from [DeepSeek Platform](https://platform.deepseek.com/) and set `DEEPSEEK_API_KEY` in the script.

## Quick Start

```bash
# 1. Download audio from YouTube
python YouTubeToWav/youtube_to_wav.py "https://www.youtube.com/watch?v=..."

# 2. Transcribe audio to text
python transcribe_wav.py

# 3. Add punctuation and formatting
python batch_punctuate.py --stream
```

## Technologies

Python, faster-whisper, yt-dlp, CUDA, DeepSeek API, FFmpeg
