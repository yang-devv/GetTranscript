"""
stats_transcripts.py

Print transcript file lengths from transcripts_wav/, ranked by character count.

Usage:
    python stats_transcripts.py           # longest first (default)
    python stats_transcripts.py --asc     # shortest first
"""

import os
import glob
import argparse

INPUT_DIR = "transcripts_wav"
TOP_N = 10

def main():
    parser = argparse.ArgumentParser(description="Show transcript character counts")
    parser.add_argument("--asc", action="store_true", help="Sort shortest first (default: longest first)")
    args = parser.parse_args()

    txt_files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    if not txt_files:
        print(f"No .txt files found in '{INPUT_DIR}'.")
        return

    results = []
    for path in txt_files:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        results.append((os.path.basename(path), len(content)))

    descending = not args.asc
    results.sort(key=lambda x: x[1], reverse=descending)
    order_label = "longest first" if descending else "shortest first"

    print(f"Order: {order_label}  |  Top {TOP_N}")
    print(f"{'Rank':<5} {'Chars':>8}  {'Filename'}")
    print("-" * 60)
    for rank, (name, length) in enumerate(results[:TOP_N], 1):
        print(f"{rank:<5} {length:>8,}  {name}")

    print("-" * 60)
    print(f"Total: {len(results)} files, {sum(x[1] for x in results):,} characters")

if __name__ == "__main__":
    main()
