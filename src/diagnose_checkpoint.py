# -*- coding: utf-8 -*-
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import config  # noqa: E402
from dataset import HandwritingDataset  # noqa: E402
from generate import generate  # noqa: E402
from model import HandwritingSynthesis  # noqa: E402
from train import batch_losses, collate_fn  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def parse_ints(raw):
    if not raw.strip():
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_texts(raw):
    if not raw:
        return list(config.target_texts)
    path = Path(raw)
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [part.strip() for part in raw.split(",") if part.strip()]


def dataset_ids(text_dir):
    ids = []
    for path in Path(text_dir).glob("trajectory_*.txt"):
        match = re.fullmatch(r"trajectory_(\d+)\.txt", path.name)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def sample_index_by_id(text_dir):
    return {item_id: idx for idx, item_id in enumerate(dataset_ids(text_dir))}


def denormalize_dxy(dxy, mean, std):
    return dxy * std.reshape(1, 2) + mean.reshape(1, 2)


def absolute_points_from_dxy(dxy, pen):
    xy = np.cumsum(dxy, axis=0)
    return np.concatenate([xy, pen.reshape(-1, 1)], axis=1)


def draw_trajectory(ax, trajectory, title):
    xs, ys = [], []
    for x, y, e in trajectory:
        xs.append(float(x))
        ys.append(float(y))
        if int(e) == 1:
            if len(xs) > 1:
                ax.plot(xs, ys, "k-", linewidth=0.8)
            xs, ys = [], []
    if len(xs) > 1:
        ax.plot(xs, ys, "k-", linewidth=0.8)
    ax.invert_yaxis()
    ax.axis("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9)


@torch.no_grad()
def teacher_forced_prediction(model, sample, checkpoint_mean, checkpoint_std):
    loader = DataLoader([sample], batch_size=1, collate_fn=collate_fn)
    batch = next(iter(loader))
    losses = batch_losses(model, batch, config.device)

    dxy = batch["dxy"].to(config.device)
    e_target = batch["e"].to(config.device)
    text = batch["text"].to(config.device)
    text_lengths = batch["text_lengths"].to(config.device)
    outputs = model(dxy, e_target, text, text_lengths, teacher_forcing_ratio=1.0)

    pi = outputs["pi"][0]
    mu = outputs["mu"][0]
    mixture_idx = pi.argmax(dim=1)
    pred_norm = mu[torch.arange(mu.shape[0], device=mu.device), mixture_idx].detach().cpu().numpy()
    target_norm = dxy[0].detach().cpu().numpy()
    pen = e_target[0].detach().cpu().numpy().reshape(-1)

    pred_dxy = denormalize_dxy(pred_norm, checkpoint_mean, checkpoint_std)
    target_dxy = denormalize_dxy(target_norm, checkpoint_mean, checkpoint_std)
    pred_points = absolute_points_from_dxy(pred_dxy, pen)
    target_points = absolute_points_from_dxy(target_dxy, pen)

    kappa = outputs["kappa"][0].detach().cpu().numpy()
    kappa_last = kappa[-1].tolist() if len(kappa) else []

    return {
        "target_points": target_points,
        "pred_points": pred_points,
        "losses": {key: float(value.item()) for key, value in losses.items()},
        "kappa_last_mean": float(np.mean(kappa_last)) if kappa_last else 0.0,
        "kappa_last_max": float(np.max(kappa_last)) if kappa_last else 0.0,
        "kappa_target": int(text_lengths[0].item()),
    }


def save_reconstruction_plot(path, target_points, pred_points, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    draw_trajectory(axes[0], target_points, "target")
    draw_trajectory(axes[1], pred_points, "teacher-forced pred")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_generation_plot(path, trajectory, title):
    fig, ax = plt.subplots(figsize=(7, 2.5))
    draw_trajectory(ax, trajectory, title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Diagnose checkpoint learning vs free generation.")
    parser.add_argument("--checkpoints", default=config.checkpoints)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--dataset", default=config.data_path)
    parser.add_argument("--text-dir", default=str(PROJECT_ROOT / "dataset" / "texts"))
    parser.add_argument("--sample-ids", default="1,2,3")
    parser.add_argument("--texts", default="")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs" / "diagnostics" / "latest"))
    parser.add_argument("--sampling-mode", choices=["argmax", "mean", "sample"], default="argmax")
    parser.add_argument("--stop-strategy", choices=["max", "mean", "min"], default="mean")
    parser.add_argument("--min-len-per-char", type=int, default=45)
    parser.add_argument("--max-len-per-char", type=int, default=350)
    parser.add_argument("--bias", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = find_checkpoint(args.checkpoints, args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)

    model = make_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    checkpoint_mean = np.asarray(checkpoint.get("dxdy_mean", [[0.0, 0.0]]), dtype=np.float32)
    checkpoint_std = np.asarray(checkpoint.get("dxdy_std", [[1.0, 1.0]]), dtype=np.float32)

    dataset = HandwritingDataset(
        args.dataset,
        max_seq_len=config.max_seq_len,
        cache_prepared=False,
    )
    dataset.dxdy_mean = checkpoint_mean
    dataset.dxdy_std = checkpoint_std

    id_to_index = sample_index_by_id(args.text_dir)
    sample_ids = parse_ints(args.sample_ids)
    texts = parse_texts(args.texts)

    report = {
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "sampling_mode": args.sampling_mode,
        "reconstructions": [],
        "generations": [],
    }

    for sample_id in sample_ids:
        if sample_id not in id_to_index:
            continue
        sample = dataset[id_to_index[sample_id]]
        text_path = Path(args.text_dir) / f"trajectory_{sample_id}.txt"
        label = text_path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n").strip()
        result = teacher_forced_prediction(model, sample, checkpoint_mean, checkpoint_std)
        png_path = output_dir / f"reconstruct_{sample_id}.png"
        save_reconstruction_plot(
            png_path,
            result["target_points"],
            result["pred_points"],
            f"trajectory_{sample_id}: {label}",
        )
        report["reconstructions"].append(
            {
                "sample_id": sample_id,
                "text": label,
                "png_path": str(png_path),
                "losses": result["losses"],
                "kappa_last_mean": result["kappa_last_mean"],
                "kappa_last_max": result["kappa_last_max"],
                "kappa_target": result["kappa_target"],
            }
        )

    for text in texts:
        min_len = max(1, len(text) * args.min_len_per_char)
        max_len = max(min_len + 1, len(text) * args.max_len_per_char)
        trajectory, diagnostics = generate(
            model,
            text,
            config.char_to_idx,
            max_len=max_len,
            device=config.device,
            bias=args.bias,
            dxdy_mean=checkpoint_mean,
            dxdy_std=checkpoint_std,
            min_len=min_len,
            append_eos=getattr(config, "append_eos_to_generation", True),
            eos_char=config.eos_char,
            stop_strategy=args.stop_strategy,
            sampling_mode=args.sampling_mode,
            progress=False,
            return_diagnostics=True,
        )
        safe = re.sub(r"[^\wа-яА-ЯёЁ]+", "_", text, flags=re.UNICODE).strip("_") or "text"
        png_path = output_dir / f"generate_{safe}_{args.sampling_mode}.png"
        json_path = output_dir / f"generate_{safe}_{args.sampling_mode}.json"
        save_generation_plot(png_path, trajectory, f"{text} | {args.sampling_mode}")
        json_path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8")
        report["generations"].append(
            {
                "text": text,
                "png_path": str(png_path),
                "json_path": str(json_path),
                "diagnostics": diagnostics,
            }
        )

    report_path = output_dir / "diagnostics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Checkpoint:", checkpoint_path)
    print("Saved:", report_path)
    for item in report["reconstructions"]:
        print(
            f"reconstruct {item['sample_id']}: "
            f"loss={item['losses']['loss']:.4f} "
            f"kappa_mean={item['kappa_last_mean']:.2f}/{item['kappa_target']} "
            f"png={item['png_path']}"
        )
    for item in report["generations"]:
        diag = item["diagnostics"]
        print(
            f"generate {item['text']!r}: "
            f"points={diag['points']} finished={diag['finished']} "
            f"attention={diag['attention_progress']:.2f}/{diag['kappa_target']} "
            f"png={item['png_path']}"
        )


if __name__ == "__main__":
    main()
