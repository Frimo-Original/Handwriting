import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config
from model import HandwritingSynthesis
from utils import sample_mdn


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_paths = []
    for path in Path(checkpoint_dir).glob("epoch_*.pth"):
        match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
        if match:
            checkpoint_paths.append((int(match.group(1)), path))
    if not checkpoint_paths:
        return None
    return max(checkpoint_paths, key=lambda item: item[0])[1]


@torch.no_grad()
def generate(
        model,
        text,
        char_to_idx,
        max_len,
        device,
        bias=0.0,
        dxdy_mean=None,
        dxdy_std=None,
        min_len=0,
        append_eos=False,
        eos_char="\n",
        stop_strategy="max",
        progress=True,
        return_diagnostics=False,
):
    model.eval()
    conditioned_text = text + eos_char if append_eos and eos_char in char_to_idx else text
    indices = [char_to_idx.get(ch, char_to_idx[" "]) for ch in conditioned_text]
    text_tensor = torch.tensor([indices], device=device, dtype=torch.long)
    char_emb = model.text_embed(text_tensor)

    lstm_size = model.decoder.lstm_size
    h1 = torch.zeros(1, lstm_size, device=device)
    c1 = torch.zeros(1, lstm_size, device=device)
    h2 = torch.zeros(1, lstm_size, device=device)
    c2 = torch.zeros(1, lstm_size, device=device)
    h3 = torch.zeros(1, lstm_size, device=device)
    c3 = torch.zeros(1, lstm_size, device=device)
    kappa = torch.zeros(1, model.decoder.K, device=device)
    w_prev = torch.zeros(1, char_emb.shape[-1], device=device)

    prev_x = torch.zeros(1, 2, device=device)
    prev_e = torch.zeros(1, 1, device=device)
    points = []
    kappa_history = []
    e_prob_history = []
    selected_pi_history = []
    x_abs, y_abs = 0.0, 0.0

    iterator = tqdm(range(max_len), desc="Generating", disable=not progress)
    for step in iterator:
        (
            e_logit,
            pi,
            mu,
            sigma,
            rho,
            h1,
            c1,
            h2,
            c2,
            h3,
            c3,
            w_prev,
            kappa,
        ) = model.decoder(
            x_t=prev_x,
            e_prev=prev_e,
            char_embeddings=char_emb,
            h1=h1,
            c1=c1,
            h2=h2,
            c2=c2,
            h3=h3,
            c3=c3,
            kappa_prev=kappa,
            w_prev=w_prev,
        )

        if bias > 0:
            pi = F.softmax(torch.log(torch.clamp(pi, min=1e-8)) * (1.0 + bias), dim=-1)
            sigma = sigma * math.exp(-bias)

        sample = sample_mdn(pi, mu, sigma, rho)
        prev_x = sample

        sample_np = sample.squeeze(0).cpu().numpy()
        if dxdy_mean is not None and dxdy_std is not None:
            sample_np = sample_np * dxdy_std.reshape(-1) + dxdy_mean.reshape(-1)
        dx, dy = float(sample_np[0]), float(sample_np[1])
        x_abs += dx
        y_abs += dy

        e_prob = torch.sigmoid(e_logit).item()
        e_val = int(e_prob > 0.5)
        prev_e = torch.tensor([[e_val]], device=device, dtype=torch.float32)
        points.append([x_abs, y_abs, e_val])
        kappa_history.append(kappa.squeeze(0).detach().cpu().tolist())
        e_prob_history.append(e_prob)
        selected_pi_history.append(float(pi.max().item()))

        if stop_strategy == "mean":
            progress_value = kappa.mean().item()
        elif stop_strategy == "min":
            progress_value = kappa.min().item()
        else:
            progress_value = kappa.max().item()

        if step + 1 >= min_len and progress_value > len(indices) - 0.5:
            break

    if not return_diagnostics:
        return points

    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
    else:
        bbox = [0.0, 0.0, 0.0, 0.0]

    last_kappa = kappa_history[-1] if kappa_history else []
    diagnostics = {
        "text": text,
        "conditioned_text": conditioned_text,
        "points": len(points),
        "pen_ups": sum(int(p[2]) for p in points),
        "bbox": bbox,
        "bbox_width": bbox[2] - bbox[0],
        "bbox_height": bbox[3] - bbox[1],
        "kappa_last": last_kappa,
        "kappa_last_mean": float(np.mean(last_kappa)) if last_kappa else 0.0,
        "kappa_last_max": float(np.max(last_kappa)) if last_kappa else 0.0,
        "kappa_target": len(indices),
        "finished": bool(last_kappa and max(last_kappa) > len(indices) - 0.5),
        "mean_pen_up_probability": float(np.mean(e_prob_history)) if e_prob_history else 0.0,
        "mean_max_pi": float(np.mean(selected_pi_history)) if selected_pi_history else 0.0,
    }
    return points, diagnostics


def plot_trajectory(trajectory, title):
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
    plt.show()


if __name__ == "__main__":
    checkpoint_path = find_latest_checkpoint(config.checkpoints)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No epoch_*.pth checkpoint found in {config.checkpoints}")
    output_json = "generated_trajectory.json"
    input_text = "слово"

    model = HandwritingSynthesis(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        lstm_size=config.lstm_size,
        num_layers=config.num_lstm_layers,
        K=config.K,
        n_mixtures=config.n_mixtures,
        kappa_initial_bias=config.kappa_initial_bias,
    ).to(config.device)

    ckpt = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    dxdy_mean = np.asarray(ckpt.get("dxdy_mean", [[0.0, 0.0]]), dtype=np.float32)
    dxdy_std = np.asarray(ckpt.get("dxdy_std", [[1.0, 1.0]]), dtype=np.float32)
    trajectory = generate(
        model,
        input_text,
        config.char_to_idx,
        max_len=3000,
        device=config.device,
        bias=0.5,
        dxdy_mean=dxdy_mean,
        dxdy_std=dxdy_std,
        min_len=200,
    )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(trajectory)} generated points to {output_json}")
    plot_trajectory(trajectory, f'Generated: "{input_text}"')
