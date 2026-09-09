"""
add_punctuation.py
──────────────────
给缺少标点的中文转录文本自动加标点，使用 DeepSeek Chat API。
- 支持长文：按接近上下文上限的大块分块处理
- 结果统一写入 transcripts_punctuated/ 文件夹

用法：
    python add_punctuation.py <input.txt>             # 默认（等待完整结果）
    python add_punctuation.py <input.txt> --stream    # 流式（实时看到输出）
"""

import os
import sys
import time
import argparse
from openai import OpenAI

# ─────────────────────────── 配置 ────────────────────────────────

DEEPSEEK_API_KEY  = "YOUR_API_KEY_HERE"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL             = "deepseek-chat"
OUTPUT_DIR        = "transcripts_punctuated"

# DeepSeek-chat 上下文窗口 64K token。
# 中文约 1.5 字/token，系统 prompt + 输出各留 4K token 余量。
# 安全输入上限 ≈ (64000 - 8000) * 1.5 ≈ 84000 字，取整留余量用 50000。
CHUNK_SIZE = 50_000   # 每块最大字符数
OVERLAP    = 200      # 块间重叠字符，防止断句不自然

SYSTEM_PROMPT = """你是一个中文标点符号添加助手。
用户会给你一段从语音转录来的中文文本，这段文本可能完全没有标点，或者标点很少。
请你：
1. 在合适的位置添加中文标点符号（逗号、句号、问号、感叹号、顿号、冒号、引号等）
2. 保持原文所有文字不变，不得增删任何汉字
3. 只输出加好标点的文本，不要加任何解释、说明或前缀
4. 自然地分段，每段之间用空行隔开"""

# ─────────────────────────── 工具函数 ────────────────────────────

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
            print(f"  [警告] 第 {attempt} 次调用失败: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                raise


def call_api_stream(client: OpenAI, text: str, retries: int = 3) -> str:
    """流式调用：边生成边打印 token，返回完整结果字符串。"""
    for attempt in range(1, retries + 1):
        try:
            collected = []
            # stream=True 走原始 chunk 流，每个 chunk 有标准的 choices[0].delta
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
            print()   # 换行
            return "".join(collected).strip()
        except Exception as e:
            print(f"\n  [警告] 第 {attempt} 次调用失败: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                raise


def process_file(input_path: str, stream: bool = False) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        print("输入文件为空，跳过。")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filename    = os.path.basename(input_path)
    output_path = os.path.join(OUTPUT_DIR, filename)

    mode_label = "流式" if stream else "阻塞"
    print(f"输入 : {input_path}  ({len(raw_text):,} 字)  [{mode_label}模式]")
    print(f"输出 : {output_path}")

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    chunks = chunk_text(raw_text)
    n = len(chunks)

    if n == 1:
        print(f"文本在单块范围内，一次性处理...", flush=True)
        if stream:
            print("─" * 60)
    else:
        print(f"文本较长，分为 {n} 块处理...", flush=True)

    api_fn = call_api_stream if stream else call_api

    result_parts = []
    for i, chunk in enumerate(chunks, 1):
        prefix = f"[{i}/{n}] " if n > 1 else ""
        if stream:
            if n > 1:
                print(f"\n─── 块 {i}/{n} ({len(chunk):,} 字) ───")
        else:
            print(f"  {prefix}调用 API ({len(chunk):,} 字)...", end=" ", flush=True)

        t0 = time.perf_counter()
        result_parts.append(api_fn(client, chunk))
        elapsed = time.perf_counter() - t0

        if stream:
            print(f"─── 完成 ({elapsed:.1f}s) ───")
        else:
            print(f"完成 ({elapsed:.1f}s)")

    final = "\n\n".join(result_parts)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final + "\n")

    print(f"\n✓ 结果已写入: {output_path}  ({len(final):,} 字)")


# ─────────────────────────── 入口 ────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 DeepSeek API 给中文文本自动加标点")
    parser.add_argument("input", help="输入的 .txt 文件路径")
    parser.add_argument("--stream", action="store_true", help="启用流式输出（实时打印 token）")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"错误：找不到文件 '{args.input}'")
        sys.exit(1)

    process_file(args.input, stream=args.stream)
