import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path("src").resolve()))
import config  # noqa: E402
from generate import generate  # noqa: E402
from model import HandwritingSynthesis  # noqa: E402


def env_bool(name, current):
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return current
    return raw in {"1", "true", "yes", "y", "on"}


def latest_checkpoint(checkpoint_dir):
    paths = []
    for path in Path(checkpoint_dir).glob("epoch_*.pth"):
        match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
        if match:
            paths.append((int(match.group(1)), path))
    if not paths:
        raise FileNotFoundError(f"No epoch_*.pth found in {checkpoint_dir}")
    return max(paths, key=lambda item: item[0])[1]


def save_preview(trajectory, output_png, title):
    xs, ys = [], []
    for x, y, e in trajectory:
        xs.append(x)
        ys.append(y)
        if e == 1:
            if len(xs) > 1:
                plt.plot(xs, ys, "k-", linewidth=0.8)
            xs, ys = [], []
    if len(xs) > 1:
        plt.plot(xs, ys, "k-", linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title(title)
    plt.savefig(output_png, dpi=180, bbox_inches="tight")


def main():
    config.checkpoints = str(
        Path(os.environ.get("CHECKPOINTS", config.checkpoints)).expanduser().resolve()
    )
    text = os.environ.get("TEXT", "sample")
    bias = float(os.environ.get("BIAS", "1.0"))
    min_len = int(os.environ.get("MIN_GEN_LEN", "200"))
    max_len = int(os.environ.get("MAX_GEN_LEN", "3000"))
    stop_strategy = os.environ.get("STOP_STRATEGY", "max")
    append_eos = env_bool("APPEND_EOS", getattr(config, "append_eos_to_generation", True))
    output_json = os.environ.get("OUTPUT_JSON", "generated_trajectory.json")
    output_png = os.environ.get("OUTPUT_PNG", "generated_trajectory.png")
    output_meta = os.environ.get("OUTPUT_META", "generated_trajectory.meta.json")

    checkpoint_path = latest_checkpoint(config.checkpoints)
    print("Checkpoint:", checkpoint_path)
    print("Text:", text)

    model = HandwritingSynthesis(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        lstm_size=config.lstm_size,
        num_layers=config.num_lstm_layers,
        K=config.K,
        n_mixtures=config.n_mixtures,
        kappa_initial_bias=config.kappa_initial_bias,
    ).to(config.device)

    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    dxdy_mean = np.asarray(checkpoint.get("dxdy_mean", [[0.0, 0.0]]), dtype=np.float32)
    dxdy_std = np.asarray(checkpoint.get("dxdy_std", [[1.0, 1.0]]), dtype=np.float32)

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
        append_eos=append_eos,
        eos_char=config.eos_char,
        stop_strategy=stop_strategy,
        return_diagnostics=True,
    )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False)
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)

    save_preview(trajectory, output_png, f'Generated: "{text}"')
    print(f"Saved {len(trajectory)} points to {output_json}")
    print(f"Saved diagnostics to {output_meta}")
    print(f"Saved preview to {output_png}")


if __name__ == "__main__":
    main()
