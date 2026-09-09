"""
youtube_to_wav.py

Download audio from a YouTube URL and convert to WAV.
Output filename = video title.wav, saved in output/ directory.

Usage:
    python youtube_to_wav.py <YouTube URL>
    python youtube_to_wav.py "https://www.youtube.com/watch?v=xxx"

Requirements:
    pip install yt-dlp
    (ffmpeg must be on your system PATH)
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
        # Uncomment if you hit bot detection (requires Chrome logged into YouTube)
        # "cookiesfrombrowser": ("chrome",),
    }

    print(f"Processing: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "unknown")
        print(f"\nDone! File saved to: {output_dir}/{title}.wav")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python youtube_to_wav.py <YouTube URL>")
        sys.exit(1)

    download_wav(sys.argv[1])
