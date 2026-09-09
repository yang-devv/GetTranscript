"""
stats_transcripts.py
────────────────────
统计 transcripts_wav/ 下所有 .txt 文件的内容长度，并打印前 10 名。

用法：
    python stats_transcripts.py           # 从长到短（默认）
    python stats_transcripts.py --asc     # 从短到长
"""

import os
import glob
import argparse

INPUT_DIR = "transcripts_wav"
TOP_N = 10

def main():
    parser = argparse.ArgumentParser(description="统计转录文本字数")
    parser.add_argument("--asc", action="store_true", help="从短到长排序（默认从长到短）")
    args = parser.parse_args()

    txt_files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    if not txt_files:
        print(f"在 '{INPUT_DIR}' 中没有找到 .txt 文件。")
        return

    results = []
    for path in txt_files:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        results.append((os.path.basename(path), len(content)))

    descending = not args.asc
    results.sort(key=lambda x: x[1], reverse=descending)
    order_label = "从长到短" if descending else "从短到长"

    print(f"排序: {order_label}  |  前 {TOP_N} 名")
    print(f"{'排名':<5} {'字数':>8}  {'文件名'}")
    print("─" * 60)
    for rank, (name, length) in enumerate(results[:TOP_N], 1):
        print(f"{rank:<5} {length:>8,}  {name}")

    print("─" * 60)
    print(f"共 {len(results)} 个文件，总字数 {sum(x[1] for x in results):,}")

if __name__ == "__main__":
    main()
