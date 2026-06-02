# -*- coding: utf-8 -*-
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import config
from generate import generate
from model import HandwritingSynthesis


def parse_texts(raw):
    if not raw:
        return list(config.target_texts)
    path = Path(raw)
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [part.strip() for part in raw.split(",") if part.strip()]


def find_checkpoint(checkpoint_dir, checkpoint):
    if checkpoint:
        return Path(checkpoint).expanduser().resolve()

    paths = []
    for path in Path(checkpoint_dir).glob("epoch_*.pth"):
        match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
        if match:
            paths.append((int(match.group(1)), path))
    if paths:
        return max(paths, key=lambda item: item[0])[1]

    best = Path(checkpoint_dir) / "best.pth"
    if best.exists():
        return best

    raise FileNotFoundError(f"No epoch_*.pth or best.pth found in {checkpoint_dir}")


def safe_name(text, max_len=48):
    cleaned = re.sub(r"[^\wа-яА-ЯёЁ]+", "_", text, flags=re.UNICODE).strip("_")
    return (cleaned[:max_len] or "text").lower()


def draw_trajectory(ax, trajectory):
    xs, ys = [], []
    for x, y, e in trajectory:
        xs.append(x)
        ys.append(y)
        if e == 1:
            if len(xs) > 1:
                ax.plot(xs, ys, "k-", linewidth=0.8)
            xs, ys = [], []
    if len(xs) > 1:
        ax.plot(xs, ys, "k-", linewidth=0.8)
    ax.invert_yaxis()
    ax.axis("equal")
    ax.axis("off")


def save_preview(path, trajectory, title):
    fig, ax = plt.subplots(figsize=(7, 2.5))
    draw_trajectory(ax, trajectory)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def candidate_score(text, diagnostics, min_len, max_len):
    points = diagnostics["points"]
    width = max(diagnostics["bbox_width"], 1e-6)
    height = max(diagnostics["bbox_height"], 1e-6)
    aspect = width / height

    score = 100.0
    score -= min(abs(points - (min_len + max_len) / 2.0) / max(max_len, 1) * 25.0, 25.0)
    score -= 35.0 if not diagnostics["finished"] else 0.0
    score -= min(abs(diagnostics["kappa_last_max"] - diagnostics["kappa_target"]) * 8.0, 35.0)
    score -= min(max(0.0, 1.2 - aspect) * 12.0, 15.0)
    score -= min(max(0.0, aspect - max(3.0, len(text) * 1.4)) * 2.0, 15.0)
    score -= min(max(0, diagnostics["pen_ups"] - max(2, len(text) + 2)) * 2.0, 15.0)
    score += min(diagnostics["mean_max_pi"] * 6.0, 6.0)
    return round(score, 4)


def make_model():
    return HandwritingSynthesis(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        lstm_size=config.lstm_size,
        num_layers=config.num_lstm_layers,
        K=config.K,
        n_mixtures=config.n_mixtures,
        kappa_initial_bias=config.kappa_initial_bias,
    ).to(config.device)


def main():
    parser = argparse.ArgumentParser(description="Generate and rank handwriting candidates for target words.")
    parser.add_argument("--texts", default="", help="Comma-separated texts or path to one text per line.")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path. Defaults to latest epoch_*.pth.")
    parser.add_argument("--checkpoints", default=config.checkpoints)
    parser.add_argument("--output-dir", default=config.candidate_output_dir)
    parser.add_argument("--biases", default=",".join(str(v) for v in config.candidate_biases))
    parser.add_argument("--variants", type=int, default=config.candidate_variants_per_bias)
    parser.add_argument("--base-seed", type=int, default=config.candidate_base_seed)
    parser.add_argument(
        "--append-eos",
        action=argparse.BooleanOptionalAction,
        default=getattr(config, "append_eos_to_generation", True),
    )
    parser.add_argument("--stop-strategy", choices=["max", "mean", "min"], default="max")
    args = parser.parse_args()

    texts = parse_texts(args.texts)
    biases = [float(value.strip()) for value in args.biases.split(",") if value.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = find_checkpoint(args.checkpoints, args.checkpoint)
    print("Checkpoint:", checkpoint_path)
    print("Device:", config.device)
    print("Texts:", texts)

    model = make_model()
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    dxdy_mean = np.asarray(checkpoint.get("dxdy_mean", [[0.0, 0.0]]), dtype=np.float32)
    dxdy_std = np.asarray(checkpoint.get("dxdy_std", [[1.0, 1.0]]), dtype=np.float32)

    rows = []
    best_by_text = {}

    for text_idx, text in enumerate(texts):
        text_dir = output_dir / f"{text_idx + 1:02d}_{safe_name(text)}"
        text_dir.mkdir(parents=True, exist_ok=True)
        min_len = max(1, len(text) * int(config.candidate_min_len_per_char))
        max_len = max(min_len + 1, len(text) * int(config.candidate_max_len_per_char))

        for bias_idx, bias in enumerate(biases):
            for variant in range(args.variants):
                seed = args.base_seed + text_idx * 100000 + bias_idx * 1000 + variant
                torch.manual_seed(seed)
                if config.device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)

                trajectory, diagnostics = generate(
                    model,
                    text,
                    config.char_to_idx,
                    max_len=max_len,
                    device=config.device,
                    bias=bias,
                    dxdy_mean=dxdy_mean,
                    dxdy_std=dxdy_std,
                    min_len=min_len,
                    append_eos=args.append_eos,
                    eos_char=config.eos_char,
                    stop_strategy=args.stop_strategy,
                    progress=False,
                    return_diagnostics=True,
                )
                score = candidate_score(text, diagnostics, min_len, max_len)
                stem = f"bias_{str(bias).replace('.', '_')}_seed_{seed}"
                json_path = text_dir / f"{stem}.json"
                meta_path = text_dir / f"{stem}.meta.json"
                png_path = text_dir / f"{stem}.png"

                json_path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8")
                meta = {"score": score, "bias": bias, "seed": seed, **diagnostics}
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                save_preview(png_path, trajectory, f'{text} | score={score:.1f} bias={bias:g}')

                row = {
                    "text": text,
                    "score": score,
                    "bias": bias,
                    "seed": seed,
                    "points": diagnostics["points"],
                    "pen_ups": diagnostics["pen_ups"],
                    "finished": diagnostics["finished"],
                    "kappa_last_max": diagnostics["kappa_last_max"],
                    "kappa_target": diagnostics["kappa_target"],
                    "json_path": str(json_path),
                    "png_path": str(png_path),
                    "meta_path": str(meta_path),
                }
                rows.append(row)
                if text not in best_by_text or score > best_by_text[text]["score"]:
                    best_by_text[text] = row
                print(f"{text!r} bias={bias:g} seed={seed} score={score:.1f} points={diagnostics['points']}")

    csv_path = output_dir / "candidates.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["text"], -row["score"])))

    best_path = output_dir / "best_candidates.json"
    best_path.write_text(json.dumps(best_by_text, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", csv_path)
    print("Saved:", best_path)


if __name__ == "__main__":
    main()
