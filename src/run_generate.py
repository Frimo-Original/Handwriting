import json
import re
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from generate import generate
from model import HandwritingSynthesis


CHECKPOINT_PATH = None

# INPUT_TEXT = "пример"
with open("../dataset/texts/trajectory_160.txt", "r", encoding="utf-8") as f:
    INPUT_TEXT = f.read()
    print(INPUT_TEXT)

MAX_GEN_LEN = 5000
MIN_GEN_LEN = 1
BIAS = 1.25
VARIANTS = 6
BASE_SEED = None  # Set an integer for repeatable variants, for example 123.
OUTPUT_DIR = Path("generated_variants")
OUTPUT_PREFIX = "generated_trajectory"
SAVE_PREVIEW = True
PREVIEW_PATH = OUTPUT_DIR / "preview.png"


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_paths = []
    for path in Path(checkpoint_dir).glob("epoch_*.pth"):
        match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
        if match:
            checkpoint_paths.append((int(match.group(1)), path))
    if not checkpoint_paths:
        return None
    return max(checkpoint_paths, key=lambda item: item[0])[1]


def draw_trajectory(ax, trajectory, title):
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
    ax.set_title(title, fontsize=10)


def plot_variants(trajectories, text):
    cols = 2
    rows = math.ceil(len(trajectories) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), squeeze=False)
    fig.suptitle(f'Generated variants: "{text}"', fontsize=14)

    for ax in axes.ravel():
        ax.axis("off")

    for idx, trajectory in enumerate(trajectories, start=1):
        ax = axes.ravel()[idx - 1]
        draw_trajectory(ax, trajectory, f"Variant {idx}: {len(trajectory)} points")

    fig.tight_layout()
    if SAVE_PREVIEW:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW_PATH, dpi=180, bbox_inches="tight")
        print(f"Saved preview to {PREVIEW_PATH}")
    plt.show()


model = HandwritingSynthesis(
    vocab_size=config.vocab_size,
    embed_dim=config.embed_dim,
    lstm_size=config.lstm_size,
    num_layers=config.num_lstm_layers,
    K=config.K,
    n_mixtures=config.n_mixtures,
    kappa_initial_bias=config.kappa_initial_bias,
).to(config.device)

checkpoint_path = Path(CHECKPOINT_PATH) if CHECKPOINT_PATH else find_latest_checkpoint(config.checkpoints)
if checkpoint_path is None:
    raise FileNotFoundError(f"No epoch_*.pth checkpoint found in {config.checkpoints}")

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

dxdy_mean = np.asarray(checkpoint.get("dxdy_mean", [[0.0, 0.0]]), dtype=np.float32)
dxdy_std = np.asarray(checkpoint.get("dxdy_std", [[1.0, 1.0]]), dtype=np.float32)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
all_trajectories = []

for variant_idx in range(1, VARIANTS + 1):
    if BASE_SEED is not None:
        seed = BASE_SEED + variant_idx - 1
        torch.manual_seed(seed)
        if config.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    print(f"\nGenerating variant {variant_idx}/{VARIANTS}")
    trajectory = generate(
        model,
        INPUT_TEXT,
        config.char_to_idx,
        max_len=MAX_GEN_LEN,
        device=config.device,
        bias=BIAS,
        dxdy_mean=dxdy_mean,
        dxdy_std=dxdy_std,
        min_len=MIN_GEN_LEN,
    )
    all_trajectories.append(trajectory)

    output_json = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{variant_idx:02d}.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(trajectory)} generated points to {output_json}")

combined_json = OUTPUT_DIR / f"{OUTPUT_PREFIX}_all.json"
with open(combined_json, "w", encoding="utf-8") as f:
    json.dump(all_trajectories, f, indent=2, ensure_ascii=False)
print(f"Saved all variants to {combined_json}")

plot_variants(all_trajectories, INPUT_TEXT)
