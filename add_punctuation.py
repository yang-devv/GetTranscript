"""
add_punctuation.py

Add punctuation to a single Chinese transcript file using the DeepSeek Chat API.
Supports long text via chunked processing. Results saved to transcripts_punctuated/.

Usage:
    python add_punctuation.py <input.txt>             # blocking (wait for full response)
    python add_punctuation.py <input.txt> --stream    # streaming (see output in real time)
"""

import os
import sys
import time
import argparse
from openai import OpenAI

# ----------------------------- Config --------------------------------

DEEPSEEK_API_KEY  = "YOUR_API_KEY_HERE"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL             = "deepseek-chat"
OUTPUT_DIR        = "transcripts_punctuated"

# DeepSeek-chat context window: 64K tokens.
# Chinese ~1.5 chars/token, reserving 4K tokens each for system prompt + output.
# Safe input limit ~ (64000 - 8000) * 1.5 ~ 84000 chars, rounded down to 50000.
CHUNK_SIZE = 50_000
OVERLAP    = 200

SYSTEM_PROMPT = """You are a Chinese punctuation assistant.
The user will give you a Chinese text transcribed from speech, with little or no punctuation.
Please:
1. Add appropriate Chinese punctuation marks (commas, periods, question marks, exclamation marks, enumeration commas, colons, quotation marks, etc.)
2. Keep all original characters unchanged — do not add or remove any Chinese characters
3. Output only the punctuated text, with no explanations or prefixes
4. Break into natural paragraphs separated by blank lines"""

# ----------------------------- Helpers -------------------------------

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
            print(f"  [warn] API call attempt {attempt} failed: {e}")
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
            print(f"\n  [warn] API call attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                raise


def process_file(input_path: str, stream: bool = False) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        print("Input file is empty, skipping.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filename    = os.path.basename(input_path)
    output_path = os.path.join(OUTPUT_DIR, filename)

    mode_label = "streaming" if stream else "blocking"
    print(f"Input : {input_path}  ({len(raw_text):,} chars)  [{mode_label} mode]")
    print(f"Output: {output_path}")

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    chunks = chunk_text(raw_text)
    n = len(chunks)

    if n == 1:
        print("Text fits in a single chunk, processing...", flush=True)
        if stream:
            print("-" * 60)
    else:
        print(f"Text is long, splitting into {n} chunks...", flush=True)

    api_fn = call_api_stream if stream else call_api

    result_parts = []
    for i, chunk in enumerate(chunks, 1):
        prefix = f"[{i}/{n}] " if n > 1 else ""
        if stream:
            if n > 1:
                print(f"\n--- chunk {i}/{n} ({len(chunk):,} chars) ---")
        else:
            print(f"  {prefix}Calling API ({len(chunk):,} chars)...", end=" ", flush=True)

        t0 = time.perf_counter()
        result_parts.append(api_fn(client, chunk))
        elapsed = time.perf_counter() - t0

        if stream:
            print(f"--- done ({elapsed:.1f}s) ---")
        else:
            print(f"done ({elapsed:.1f}s)")

    final = "\n\n".join(result_parts)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final + "\n")

    print(f"\nResult written to: {output_path}  ({len(final):,} chars)")


# ------------------------------ Entry --------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add punctuation to Chinese transcript using DeepSeek API")
    parser.add_argument("input", help="Input .txt file path")
    parser.add_argument("--stream", action="store_true", help="Enable streaming output (print tokens in real time)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: file not found '{args.input}'")
        sys.exit(1)

    process_file(args.input, stream=args.stream)
