"""
batch_punctuate.py
──────────────────
批量处理 transcripts_wav/ 下所有 .txt 文件，
使用 DeepSeek API 加标点，结果保存到 transcripts_punctuated/。
已处理过的文件自动跳过（断点续跑）。

用法：
    python batch_punctuate.py             # 默认（阻塞，等完整结果）
    python batch_punctuate.py --stream    # 流式（实时打印 token）
"""

import os
import sys
import glob
import time
import argparse
from datetime import datetime
from openai import OpenAI

# ─────────────────────────── 配置 ────────────────────────────────

DEEPSEEK_API_KEY  = "YOUR_API_KEY_HERE"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL             = "deepseek-chat"

INPUT_DIR  = "transcripts_wav"
OUTPUT_DIR = "transcripts_punctuated"

CHUNK_SIZE = 50_000   # 每块最大字符数（DeepSeek 64K token 安全上限）
OVERLAP    = 200      # 块间重叠字符，防止断句不自然

SYSTEM_PROMPT = """你是一个中文标点符号添加助手。
用户会给你一段从语音转录来的中文文本，这段文本可能完全没有标点，或者标点很少。
请你：
1. 在合适的位置添加中文标点符号（逗号、句号、问号、感叹号、顿号、冒号、引号等）
2. 保持原文所有文字不变，不得增删任何汉字
3. 只输出加好标点的文本，不要加任何解释、说明或前缀
4. 自然地分段，每段之间用空行隔开"""

# ─────────────────────────── 工具函数 ────────────────────────────

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
    """仅在文本超过 CHUNK_SIZE 时才分块，否则直接返回整段。"""
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
    """阻塞调用：等待完整响应后返回。"""
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
            log(f"第 {attempt} 次 API 调用失败: {e}", "WARN")
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                raise


def call_api_stream(client: OpenAI, text: str, retries: int = 3) -> str:
    """流式调用：边生成边打印 token，返回完整结果字符串。"""
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
            print()  # 换行
            return "".join(collected).strip()
        except Exception as e:
            log(f"\n第 {attempt} 次 API 调用失败: {e}", "WARN")
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                raise


def process_one(client: OpenAI, input_path: str, output_path: str, stream: bool = False) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        log(f"文件为空，跳过: {os.path.basename(input_path)}", "WARN")
        return

    chunks = chunk_text(raw_text)
    n = len(chunks)
    chunk_info = f"共 {n} 块" if n > 1 else "单块"
    mode_label = "流式" if stream else "阻塞"
    log(f"处理中 ({len(raw_text):,} 字, {chunk_info}, {mode_label}): {os.path.basename(input_path)}")

    api_fn = call_api_stream if stream else call_api

    result_parts = []
    for i, chunk in enumerate(chunks, 1):
        prefix = f"块 {i}/{n} " if n > 1 else ""
        log(f"  {prefix}调用 API ({len(chunk):,} 字)...")
        t0 = time.perf_counter()
        result_parts.append(api_fn(client, chunk))
        log(f"  {prefix}完成 ({time.perf_counter() - t0:.1f}s)")

    final = "\n\n".join(result_parts)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final + "\n")


# ─────────────────────────── 主流程 ──────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="批量给中文文本加标点")
    parser.add_argument("--stream", action="store_true", help="启用流式输出（实时打印 token）")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    txt_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
    total = len(txt_files)

    if total == 0:
        log(f"在 '{INPUT_DIR}' 中没有找到 .txt 文件。")
        return

    mode_label = "流式" if args.stream else "阻塞"
    log(f"找到 {total} 个 .txt 文件，开始批量处理（{mode_label}模式）...")
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    succeeded = 0
    skipped   = 0
    failed    = 0
    session_start = time.perf_counter()

    for i, input_path in enumerate(txt_files, 1):
        filename    = os.path.basename(input_path)
        output_path = os.path.join(OUTPUT_DIR, filename)

        # 断点续跑：已存在则跳过
        if os.path.exists(output_path):
            log(f"[{i}/{total}] 跳过（已存在）: {filename}")
            skipped += 1
            continue

        log(f"[{i}/{total}] {filename}")
        t_file = time.perf_counter()
        try:
            process_one(client, input_path, output_path, stream=args.stream)
            elapsed = time.perf_counter() - t_file
            log(f"[{i}/{total}] ✓ 完成 {filename}  耗时 {elapsed:.1f}s")
            succeeded += 1
        except Exception as e:
            elapsed = time.perf_counter() - t_file
            log(f"[{i}/{total}] ✗ 失败 {filename}  耗时 {elapsed:.1f}s  错误: {e}", "ERROR")
            failed += 1

    total_elapsed = time.perf_counter() - session_start
    m, s = divmod(int(total_elapsed), 60)
    log("─" * 60)
    log(f"全部完成 | 总耗时 {m}m{s:02d}s | 成功 {succeeded} | 跳过 {skipped} | 失败 {failed}")


if __name__ == "__main__":
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== 批量加标点任务开始 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    main()
