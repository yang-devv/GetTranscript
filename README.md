# GetTranscript

GetTranscript 是一个功能强大的本地化音视频转录与文本处理工具箱。它的主要目的是将视频或音频文件（如 YouTube 视频或本地 `.wav`/`.wma` 音频）转换为高质量、排版整洁的中文逐字稿，并且能够智能修复无标点符号的中文长文本。

## 🌟 核心功能

本项目主要包含以下三个独立的自动化功能模块：

1. **YouTube 视频 WAV 自动提取** (`youtube_to_wav.py`)
   - 只需要提供一个 YouTube 视频或播放列表链接，脚本会自动通过 `yt-dlp` 将其下载并提取为最高质量的 `.wav` 音频文件。
2. **WAV/音频转字幕与纯文本** (`transcribe_wav.py` / `transcribe.py`)
   - 利用强悍的本地开源大模型（基于 `faster-whisper` 的 `large-v3`）进行语音识别，将音频转换成原始文本。
   - 支持断点续传、GPU 批量加速推理，显著提升识别效率。
3. **杂乱无句读中文处理** (`batch_punctuate.py`)
   - 针对语音识别出来的一整坨“无标点、无排版”的生肉文本，调用大语言模型（如 DeepSeek API）进行智能断句、加标点符号并自然分段，最后输出排版精美、可读性极强的成品逐字稿。

---

## 🛠️ 环境配置

推荐使用 `conda` 来隔离并管理项目环境。

### 1. 创建并激活 Conda 环境
打开终端，运行以下命令：
```bash
conda create -n whisper_env python=3.10 -y
conda activate whisper_env
```

### 2. 安装 Python 依赖包
使用仓库中提供的 `requirements.txt`：
```bash
pip install -r requirements.txt
```

### 3. 安装 FFmpeg（必须）
音频处理需要底层系统安装有 FFmpeg 工具：
- **Windows**: `winget install FFmpeg` 或通过 `scoop install ffmpeg`
- **Mac**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

---

## 💻 硬件配置与 CUDA 踩坑指南

### 显存（VRAM）要求
本项目默认开启了 `faster-whisper` 的 `large-v3` 模型进行语音识别，该模型精度极高但需要一定的显存支撑：
- **最佳体验**：推荐拥有至少 **8GB 显存** 的 NVIDIA 显卡（如 RTX 3070/4060 及以上），在 `float16` 计算精度下可流畅运行。
- **降级方案**：如果您的显存为 4GB~6GB，可以将代码中的 `compute_type` 修改为 `int8_float16` 或 `int8` 来节省显存。
- **无独立显卡**：如果没有 N 卡，代码将自动 fallback 到 CPU 模式运行（这会非常慢，请做好心理准备）。

### 缺少 `cudnn_xxx.dll` 等 CUDA 报错问题
如果您在 Windows 平台上运行 Whisper 时遇到找不到特定 `cudnn` 的 DLL 文件（如 `cudnn_cnn_infer64_8.dll`），导致 GPU 无法启用，可以参考以下教程进行彻底配置和修复：
🔗 [How to Setup Whisper from OpenAI (Windows CUDA Guide)](https://joshuachini.com/2024/02/04/how-to-setup-whisper-from-openai/)

---

## 🔑 DeepSeek API 要求

智能加标点与分段脚本 (`batch_punctuate.py`) 依赖 DeepSeek 的 API。
1. 您需要前往 [DeepSeek 开放平台](https://platform.deepseek.com/) 注册并获取一个有效的 API Key。
2. 将您生成的 API Key 替换到 `batch_punctuate.py` 文件顶部的 `DEEPSEEK_API_KEY` 变量中。

> **注意**：脚本内置的旧 API Key 出于安全原因已被弃用或在官网删除。请务必使用您自己账户下最新生成的有效 Token，否则调用时会报错 `401 Authentication Fails`。

---

## 🚀 快速上手

1. **获取音频**：
   ```bash
   python YouTubeToWav/youtube_to_wav.py "https://www.youtube.com/watch?v=..."
   ```
   *(下载好的音频会在相应的 `output` 或 `rawwav` 目录中)*

2. **音频转纯文本**：
   ```bash
   python transcribe_wav.py
   ```
   *(产物会保存在 `transcripts_wav` 文件夹中)*

3. **智能加标点与排版**：
   ```bash
   python batch_punctuate.py --stream
   ```
   *(整理好的精美文章会保存在 `transcripts_punctuated` 文件夹中)*
