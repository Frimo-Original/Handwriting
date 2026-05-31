# -*- coding: utf-8 -*-
import csv
import html
import json
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from generate import generate
from model import HandwritingSynthesis


# If CHECKPOINT_EPOCHS is empty, the script picks recent checkpoints automatically.
CHECKPOINT_EPOCHS = [280, 300, 320, 340, 360, 380, 400, 430]
AUTO_MIN_EPOCH = 250
AUTO_EPOCH_STEP = 20
AUTO_MAX_EPOCHS = 8

BIASES = [0.5, 0.75, 1.0, 1.25, 1.5]
VARIANTS_PER_CELL = 1
BASE_SEED = 12345

TEST_TEXTS = [
    "слово",
    "мама мыла раму.",
    "это моя строка.",
    "Проверь строку.",
    "Почему строка уходит вниз?",
    'Аня сказала: "Готово!"',
]

MAX_GEN_LEN = 5000
MIN_GEN_LEN = 1

RUN_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(config.runs_dir) / "generation_eval" / RUN_NAME
SHOW_PLOTS = True
SAVE_JSON = True
SAVE_HTML = True


def find_checkpoints(checkpoint_dir):
    checkpoints = {}
    for path in Path(checkpoint_dir).glob("epoch_*.pth"):
        match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
        if match:
            checkpoints[int(match.group(1))] = path
    return dict(sorted(checkpoints.items()))


def select_epochs(checkpoints):
    if CHECKPOINT_EPOCHS:
        epochs = [epoch for epoch in CHECKPOINT_EPOCHS if epoch in checkpoints]
        missing = [epoch for epoch in CHECKPOINT_EPOCHS if epoch not in checkpoints]
        if missing:
            print(f"Warning: missing checkpoints for epochs: {missing}")
        if not epochs:
            raise FileNotFoundError(f"No configured checkpoints found: {CHECKPOINT_EPOCHS}")
        return epochs

    epochs = [
        epoch
        for epoch in checkpoints
        if epoch >= AUTO_MIN_EPOCH and epoch % AUTO_EPOCH_STEP == 0
    ]
    if checkpoints:
        latest_epoch = max(checkpoints)
        if latest_epoch >= AUTO_MIN_EPOCH and latest_epoch not in epochs:
            epochs.append(latest_epoch)

    epochs = sorted(epochs)
    if len(epochs) > AUTO_MAX_EPOCHS:
        epochs = epochs[-AUTO_MAX_EPOCHS:]
    if not epochs and checkpoints:
        epochs = [max(checkpoints)]
    return epochs


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


def set_seed(seed):
    torch.manual_seed(seed)
    if config.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def draw_trajectory(ax, trajectory):
    xs, ys = [], []
    for x, y, e in trajectory:
        xs.append(x)
        ys.append(y)
        if e == 1:
            if len(xs) > 1:
                ax.plot(xs, ys, "k-", linewidth=0.7)
            xs, ys = [], []
    if len(xs) > 1:
        ax.plot(xs, ys, "k-", linewidth=0.7)
    ax.invert_yaxis()
    ax.axis("equal")
    ax.axis("off")


def safe_name(text, max_len=36):
    cleaned = re.sub(r"[^\wа-яА-ЯёЁ]+", "_", text, flags=re.UNICODE).strip("_")
    return (cleaned[:max_len] or "text").lower()


def save_text_grid(text_idx, text, epochs, biases, results):
    cols = len(biases) * VARIANTS_PER_CELL
    rows = len(epochs)
    fig_width = max(12, cols * 3.2)
    fig_height = max(3, rows * 2.4)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
    fig.suptitle(f'Text {text_idx + 1}: "{text}"', fontsize=13)

    for ax in axes.ravel():
        ax.axis("off")

    for row, epoch in enumerate(epochs):
        for bias_idx, bias in enumerate(biases):
            for variant_idx in range(VARIANTS_PER_CELL):
                col = bias_idx * VARIANTS_PER_CELL + variant_idx
                item = results[(text_idx, epoch, bias, variant_idx)]
                ax = axes[row][col]
                draw_trajectory(ax, item["trajectory"])
                title = f"e{epoch} b{bias:g}"
                if VARIANTS_PER_CELL > 1:
                    title += f" v{variant_idx + 1}"
                title += f"\n{item['points']} pts"
                ax.set_title(title, fontsize=8)

    fig.tight_layout()
    grid_path = OUTPUT_DIR / f"text_{text_idx + 1:02d}_{safe_name(text)}.png"
    fig.savefig(grid_path, dpi=180, bbox_inches="tight")
    print(f"Saved grid: {grid_path}")
    return fig, grid_path


def write_score_template(rows):
    csv_path = OUTPUT_DIR / "scores_template.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "score",
                "text_idx",
                "text",
                "epoch",
                "bias",
                "variant",
                "points",
                "json_path",
                "notes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({"score": "", "notes": "", **row})
    print(f"Saved score template: {csv_path}")


def write_html_report(grid_paths, epochs, biases):
    html_path = OUTPUT_DIR / "index.html"
    lines = [
        "<!doctype html>",
        '<meta charset="utf-8">',
        "<title>Generation eval</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f7;color:#222}",
        "img{max-width:100%;border:1px solid #ddd;background:white;margin:12px 0}",
        "code{background:#eee;padding:2px 4px}",
        "</style>",
        "<h1>Generation eval</h1>",
        f"<p>Epochs: <code>{html.escape(str(epochs))}</code></p>",
        f"<p>Biases: <code>{html.escape(str(biases))}</code></p>",
        "<p>Оценивай глазами: читаемость, похожесть на почерк, конец строки, мусорные точки, горизонталь.</p>",
    ]
    for path in grid_paths:
        rel = path.relative_to(OUTPUT_DIR).as_posix()
        lines.append(f"<h2>{html.escape(path.stem)}</h2>")
        lines.append(f'<img src="{html.escape(rel)}" alt="{html.escape(path.stem)}">')

    html_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved HTML report: {html_path}")


def main():
    checkpoints = find_checkpoints(config.checkpoints)
    if not checkpoints:
        raise FileNotFoundError(f"No epoch_*.pth checkpoints found in {config.checkpoints}")

    epochs = select_epochs(checkpoints)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generation evaluation")
    print("Device:", config.device)
    print("Output:", OUTPUT_DIR)
    print("Epochs:", epochs)
    print("Biases:", BIASES)
    print("Texts:", len(TEST_TEXTS))

    model = make_model()
    results = {}
    score_rows = []

    for epoch in epochs:
        checkpoint_path = checkpoints[epoch]
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        dxdy_mean = np.asarray(checkpoint.get("dxdy_mean", [[0.0, 0.0]]), dtype=np.float32)
        dxdy_std = np.asarray(checkpoint.get("dxdy_std", [[1.0, 1.0]]), dtype=np.float32)

        for text_idx, text in enumerate(TEST_TEXTS):
            for bias in BIASES:
                for variant_idx in range(VARIANTS_PER_CELL):
                    seed = BASE_SEED + text_idx * 10000 + variant_idx
                    set_seed(seed)
                    print(
                        f"Generate text={text_idx + 1}/{len(TEST_TEXTS)} "
                        f"epoch={epoch} bias={bias:g} variant={variant_idx + 1}"
                    )
                    trajectory = generate(
                        model,
                        text,
                        config.char_to_idx,
                        max_len=MAX_GEN_LEN,
                        device=config.device,
                        bias=bias,
                        dxdy_mean=dxdy_mean,
                        dxdy_std=dxdy_std,
                        min_len=MIN_GEN_LEN,
                    )

                    json_path = ""
                    if SAVE_JSON:
                        text_dir = OUTPUT_DIR / f"text_{text_idx + 1:02d}_{safe_name(text)}"
                        text_dir.mkdir(parents=True, exist_ok=True)
                        json_file = (
                            text_dir
                            / f"epoch_{epoch:04d}_bias_{str(bias).replace('.', '_')}_v{variant_idx + 1:02d}.json"
                        )
                        with open(json_file, "w", encoding="utf-8") as f:
                            json.dump(trajectory, f, indent=2, ensure_ascii=False)
                        json_path = str(json_file)

                    results[(text_idx, epoch, bias, variant_idx)] = {
                        "trajectory": trajectory,
                        "points": len(trajectory),
                    }
                    score_rows.append(
                        {
                            "text_idx": text_idx + 1,
                            "text": text,
                            "epoch": epoch,
                            "bias": bias,
                            "variant": variant_idx + 1,
                            "points": len(trajectory),
                            "json_path": json_path,
                        }
                    )

    grid_paths = []
    figures = []
    for text_idx, text in enumerate(TEST_TEXTS):
        fig, grid_path = save_text_grid(text_idx, text, epochs, BIASES, results)
        grid_paths.append(grid_path)
        figures.append(fig)

    write_score_template(score_rows)
    if SAVE_HTML:
        write_html_report(grid_paths, epochs, BIASES)

    if SHOW_PLOTS:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main()
