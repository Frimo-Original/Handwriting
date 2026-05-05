import json
import re
from pathlib import Path

import numpy as np
import torch

import config
from generate import generate, plot_trajectory
from model import HandwritingSynthesis


CHECKPOINT_PATH = None

INPUT_TEXT = "а"
# with open("../dataset/texts/trajectory_36.txt", "r", encoding="utf-8") as f:
#     INPUT_TEXT = f.read()
#     print(INPUT_TEXT)

MAX_GEN_LEN = 5000
MIN_GEN_LEN = 1
BIAS = 1.5
OUTPUT_JSON = "generated_trajectory.json"


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_paths = []
    for path in Path(checkpoint_dir).glob("epoch_*.pth"):
        match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
        if match:
            checkpoint_paths.append((int(match.group(1)), path))
    if not checkpoint_paths:
        return None
    return max(checkpoint_paths, key=lambda item: item[0])[1]


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

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(trajectory, f, indent=2, ensure_ascii=False)
print(f"Saved {len(trajectory)} generated points to {OUTPUT_JSON}")

plot_trajectory(trajectory, f'Generated: "{INPUT_TEXT}"')
