# -*- coding: utf-8 -*-
import csv
from collections import defaultdict
from pathlib import Path


EVAL_DIR = None  # Example: Path("generation_eval/20260517_203000")


def latest_eval_dir():
    root = Path("generation_eval")
    candidates = [path for path in root.glob("*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError("No generation_eval/* folders found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_score(raw):
    raw = raw.strip().replace(",", ".")
    if not raw:
        return None
    return float(raw)


def main():
    eval_dir = Path(EVAL_DIR) if EVAL_DIR else latest_eval_dir()
    csv_path = eval_dir / "scores_template.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing score file: {csv_path}")

    by_pair = defaultdict(list)
    by_epoch = defaultdict(list)
    by_bias = defaultdict(list)
    scored_rows = 0

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = parse_score(row.get("score", ""))
            if score is None:
                continue
            epoch = int(row["epoch"])
            bias = float(row["bias"])
            by_pair[(epoch, bias)].append(score)
            by_epoch[epoch].append(score)
            by_bias[bias].append(score)
            scored_rows += 1

    if scored_rows == 0:
        print(f"No scores found in {csv_path}")
        print("Fill the score column first, for example from 0 to 5.")
        return

    def avg(values):
        return sum(values) / len(values)

    print(f"Scores: {csv_path}")
    print(f"Scored rows: {scored_rows}")

    print("\nBest epoch + bias:")
    ranked_pairs = sorted(
        by_pair.items(),
        key=lambda item: (avg(item[1]), len(item[1])),
        reverse=True,
    )
    for (epoch, bias), values in ranked_pairs[:15]:
        print(f"  epoch={epoch:4d} bias={bias:g}: avg={avg(values):.3f}, n={len(values)}")

    print("\nBest epochs:")
    ranked_epochs = sorted(by_epoch.items(), key=lambda item: avg(item[1]), reverse=True)
    for epoch, values in ranked_epochs:
        print(f"  epoch={epoch:4d}: avg={avg(values):.3f}, n={len(values)}")

    print("\nBest biases:")
    ranked_biases = sorted(by_bias.items(), key=lambda item: avg(item[1]), reverse=True)
    for bias, values in ranked_biases:
        print(f"  bias={bias:g}: avg={avg(values):.3f}, n={len(values)}")


if __name__ == "__main__":
    main()
