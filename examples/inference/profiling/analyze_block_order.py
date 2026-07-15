import os
import argparse
from collections import Counter

LOG_DIR = "logs/block_profile"

IGNORE_KEYS = {
    "fwd_block",
    "self_attn",
    "fc_in",
    "fc_out",
    "act",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    return parser.parse_args()


def parse_log(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    for i in range(len(lines) - 1):
        if "fwd_block" in lines[i]:
            labels = lines[i].strip().split("\t")
            values = list(map(float, lines[i+1].strip().split("\t")))
            return labels, values

    return None, None


def process_one(labels, values):
    data = dict(zip(labels, values))
    total = data["fwd_block"]

    items = []
    for k, v in data.items():
        if k in IGNORE_KEYS:
            continue
        items.append((k, v / total))

    items.sort(key=lambda x: -x[1])
    return items  # 注意这里返回 (name, ratio)


def main():
    args = parse_args()

    files = [f for f in os.listdir(LOG_DIR) if f.endswith(".log")]

    if args.model:
        files = [f for f in files if args.model in f]

    counter = Counter()

    # 👉 特殊情况收集
    ffn_rank0_files = []
    cross_rank1_files = []

    for fname in files:
        path = os.path.join(LOG_DIR, fname)
        labels, values = parse_log(path)
        if labels is None:
            continue

        items = process_one(labels, values)
        order = tuple(k for k, _ in items)

        counter[order] += 1

        # ===== 特殊情况检测 =====

        if len(items) > 0:
            if items[0][0] == "ffn":
                ffn_rank0_files.append(fname)

        if len(items) > 1:
            if items[1][0] == "cross_attn_core":
                cross_rank1_files.append(fname)

    total = sum(counter.values())

    print("==== ORDERINGS (by frequency) ====")
    for idx, (order, cnt) in enumerate(counter.most_common()):
        ratio = cnt / total if total > 0 else 0
        print(f"{idx}: count={cnt}, ratio={ratio:.3f}")
        print(f"   {order}")

    print(f"\nTotal unique orderings: {len(counter)}")
    print(f"Total logs processed: {total}")

    # ===== 输出特殊情况 =====
    print("\n==== SPECIAL CASES ====")

    print(f"\n[FFN at rank 0] count={len(ffn_rank0_files)}")
    for f in ffn_rank0_files:
        print(f)

    print(f"\n[Cross-attn at rank 1] count={len(cross_rank1_files)}")
    for f in cross_rank1_files:
        print(f)


if __name__ == "__main__":
    main()