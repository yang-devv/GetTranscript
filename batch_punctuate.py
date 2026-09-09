"""
batch_punctuate.py

Batch-process all .txt files in transcripts_wav/ — add punctuation
via the DeepSeek API and save results to transcripts_punctuated/.
Already-processed files are skipped automatically (checkpoint resume).

Usage:
    python batch_punctuate.py             # blocking mode (wait for full response)
    python batch_punctuate.py --stream    # streaming mode (print tokens in real time)
"""

import os
import sys
import glob
import time
import argparse
from datetime import datetime
from openai import OpenAI

# ----------------------------- Config --------------------------------

DEEPSEEK_API_KEY  = "YOUR_API_KEY_HERE"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL             = "deepseek-chat"

INPUT_DIR  = "transcripts_wav"
OUTPUT_DIR = "transcripts_punctuated"

CHUNK_SIZE = 50_000   # max characters per chunk (safe limit for DeepSeek 64K context)
OVERLAP    = 200      # overlap between chunks to avoid unnatural breaks

SYSTEM_PROMPT = """You are a Chinese punctuation assistant.
The user will give you a Chinese text transcribed from speech, with little or no punctuation.
Please:
1. Add appropriate Chinese punctuation marks (commas, periods, question marks, exclamation marks, enumeration commas, colons, quotation marks, etc.)
2. Keep all original characters unchanged — do not add or remove any Chinese characters
3. Output only the punctuated text, with no explanations or prefixes
4. Break into natural paragraphs separated by blank lines"""

# ----------------------------- Helpers -------------------------------

LOG_FILE = "batch_punctuate_log.txt"

def log(msg: str, level: str = "INFO") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {msg}"
    try:
        enc = sys.stdout.encoding or "utf-8"
        print(line.encode(enc, errors="replace").decode(enc))
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def chunk_text(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - OVERLAP
    return chunks


def call_api(client: OpenAI, text: str, retries: int = 3) -> str:
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text},
                ],
                temperature=0.3,
                max_tokens=8192,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            log(f"API call attempt {attempt} failed: {e}", "WARN")
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                raise


def call_api_stream(client: OpenAI, text: str, retries: int = 3) -> str:
    for attempt in range(1, retries + 1):
        try:
            collected = []
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text},
                ],
                temperature=0.3,
                max_tokens=8192,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
                    collected.append(delta)
            print()
            return "".join(collected).strip()
        except Exception as e:
            log(f"\nAPI call attempt {attempt} failed: {e}", "WARN")
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                raise


def process_one(client: OpenAI, input_path: str, output_path: str, stream: bool = False) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        log(f"Empty file, skipping: {os.path.basename(input_path)}", "WARN")
        return

    chunks = chunk_text(raw_text)
    n = len(chunks)
    chunk_info = f"{n} chunks" if n > 1 else "single chunk"
    mode_label = "streaming" if stream else "blocking"
    log(f"Processing ({len(raw_text):,} chars, {chunk_info}, {mode_label}): {os.path.basename(input_path)}")

    api_fn = call_api_stream if stream else call_api

    result_parts = []
    for i, chunk in enumerate(chunks, 1):
        prefix = f"chunk {i}/{n} " if n > 1 else ""
        log(f"  {prefix}calling API ({len(chunk):,} chars)...")
        t0 = time.perf_counter()
        result_parts.append(api_fn(client, chunk))
        log(f"  {prefix}done ({time.perf_counter() - t0:.1f}s)")

    final = "\n\n".join(result_parts)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final + "\n")


# ------------------------------ Main ---------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-add punctuation to Chinese transcripts")
    parser.add_argument("--stream", action="store_true", help="Enable streaming output (print tokens in real time)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    txt_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
    total = len(txt_files)

    if total == 0:
        log(f"No .txt files found in '{INPUT_DIR}'.")
        return

    mode_label = "streaming" if args.stream else "blocking"
    log(f"Found {total} .txt files, starting batch processing ({mode_label} mode)...")
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    succeeded = 0
    skipped   = 0
    failed    = 0
    session_start = time.perf_counter()

    for i, input_path in enumerate(txt_files, 1):
        filename    = os.path.basename(input_path)
        output_path = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(output_path):
            log(f"[{i}/{total}] Skipping (already exists): {filename}")
            skipped += 1
            continue

        log(f"[{i}/{total}] {filename}")
        t_file = time.perf_counter()
        try:
            process_one(client, input_path, output_path, stream=args.stream)
            elapsed = time.perf_counter() - t_file
            log(f"[{i}/{total}] Done {filename}  ({elapsed:.1f}s)")
            succeeded += 1
        except Exception as e:
            elapsed = time.perf_counter() - t_file
            log(f"[{i}/{total}] Failed {filename}  ({elapsed:.1f}s)  error: {e}", "ERROR")
            failed += 1

    total_elapsed = time.perf_counter() - session_start
    m, s = divmod(int(total_elapsed), 60)
    log("-" * 60)
    log(f"All done | total {m}m{s:02d}s | succeeded {succeeded} | skipped {skipped} | failed {failed}")


if __name__ == "__main__":
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== Batch punctuation started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    main()
