"""
youtube_to_wav.py
─────────────────
给定 YouTube URL，下载音频并转为 WAV 文件。
输出文件名 = 视频标题.wav，保存在 output/ 目录中。

用法：
    python youtube_to_wav.py <YouTube URL>
    python youtube_to_wav.py "https://www.youtube.com/watch?v=xxx"

依赖：
    pip install yt-dlp
    （ffmpeg 需在系统 PATH 中，你已安装）
"""

import sys
import os
import yt_dlp


OUTPUT_DIR = "output"


def download_wav(url: str, output_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        # 如果遇到 bot 检测，取消下面这行注释（需先在 Chrome 登录过 YouTube）
        # "cookiesfrombrowser": ("chrome",),
    }

    print(f"正在处理: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "unknown")
        print(f"\n✓ 完成！文件保存在: {output_dir}/{title}.wav")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python youtube_to_wav.py <YouTube URL>")
        sys.exit(1)

    download_wav(sys.argv[1])
