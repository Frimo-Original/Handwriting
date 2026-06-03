# -*- coding: utf-8 -*-
"""Tiny overfit experiment for checking free-run word generation."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

import config  # noqa: E402
from dataset import HandwritingDataset  # noqa: E402
from generate import generate  # noqa: E402
from model import HandwritingSynthesis  # noqa: E402
from train import batch_losses, collate_fn  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def safe_name(text, max_len=48):
    cleaned = re.sub(r"[^\wа-яА-ЯёЁ]+", "_", text, flags=re.UNICODE).strip("_")
    return (cleaned[:max_len] or "text").lower()


def parse_csv_ints(raw):
    if not raw.strip():
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_floats(raw):
    if not raw.strip():
        return []
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_modes(raw):
    modes = [part.strip() for part in raw.split(",") if part.strip()]
    allowed = {"argmax", "mean", "sample"}
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"Unknown generation modes: {', '.join(unknown)}")
    return modes


def read_words(words, words_file):
    if words:
        return list(dict.fromkeys(words))

    path = Path(words_file)
    if path.exists():
        loaded = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if loaded:
            return list(dict.fromkeys(loaded))

    return list(dict.fromkeys(config.target_texts))


def find_checkpoint(checkpoint_dir, checkpoint):
    if checkpoint:
        path = Path(checkpoint.replace("\\", "/")).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    preferred = Path(checkpoint_dir) / "epoch_272.pth"
    if preferred.exists():
        return preferred.resolve()

    paths = []
    for path in Path(checkpoint_dir).glob("epoch_*.pth"):
        match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
        if match:
            paths.append((int(match.group(1)), path))
    if paths:
        return max(paths, key=lambda item: item[0])[1].resolve()

    best = Path(checkpoint_dir) / "best.pth"
    if best.exists():
        return best.resolve()

    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


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


def checkpoint_config_matches(checkpoint):
    saved = checkpoint.get("config", {})
    expected = {
        "vocab_size": config.vocab_size,
        "embed_dim": config.embed_dim,
        "vocab_tokens": getattr(config, "VOCAB_TOKENS", list(config.CHAR_SET)),
        "eos_token": getattr(config, "eos_token", None),
        "lstm_size": config.lstm_size,
        "K": config.K,
        "n_mixtures": config.n_mixtures,
        "kappa_initial_bias": config.kappa_initial_bias,
    }
    return all(saved.get(key) == value for key, value in expected.items())


def dataset_records(json_dir, text_dir):
    records = []
    json_paths = sorted(
        Path(json_dir).glob("trajectory_*.json"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    for json_path in json_paths:
        sample_id = int(json_path.stem.split("_")[-1])
        text_path = Path(text_dir) / f"trajectory_{sample_id}.txt"
        if not text_path.exists():
            continue
        dataset_index = len(records)
        raw_text = text_path.read_text(encoding="utf-8")
        normalized = config.normalize_training_text(raw_text)
        records.append(
            {
                "dataset_index": dataset_index,
                "sample_id": sample_id,
                "text": normalized,
                "text_path": str(text_path),
                "json_path": str(json_path),
            }
        )
    return records


def select_records(records, words, sample_ids, match_mode, require_all_words):
    selected = []
    selected_keys = set()
    missing_words = []

    for sample_id in sample_ids:
        matched = [record for record in records if record["sample_id"] == sample_id]
        for record in matched:
            key = record["sample_id"]
            if key not in selected_keys:
                selected.append(record)
                selected_keys.add(key)

    for word in words:
        normalized_word = config.normalize_training_text(word)
        if match_mode == "contains":
            matches = [record for record in records if normalized_word in record["text"]]
        else:
            matches = [record for record in records if record["text"].strip() == normalized_word.strip()]

        if not matches:
            missing_words.append(word)
            continue

        for record in matches:
            key = record["sample_id"]
            if key not in selected_keys:
                selected.append(record)
                selected_keys.add(key)

    if require_all_words and missing_words:
        raise ValueError("Missing exact dataset samples for words: " + ", ".join(missing_words))

    return selected, missing_words


def draw_trajectory(ax, trajectory):
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


def save_generation_plot(path, trajectory, title):
    fig, ax = plt.subplots(figsize=(7, 2.5))
    draw_trajectory(ax, trajectory)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def checkpoint_payload(epoch, model, optimizer, dataset, records, metrics, source_checkpoint):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "dxdy_mean": dataset.dxdy_mean,
        "dxdy_std": dataset.dxdy_std,
        "metrics": metrics,
        "source_checkpoint": str(source_checkpoint) if source_checkpoint else None,
        "overfit_records": records,
        "checkpoint_type": "overfit",
        "config": {
            "vocab_size": config.vocab_size,
            "embed_dim": config.embed_dim,
            "vocab_tokens": getattr(config, "VOCAB_TOKENS", list(config.CHAR_SET)),
            "eos_token": getattr(config, "eos_token", None),
            "lstm_size": config.lstm_size,
            "K": config.K,
            "n_mixtures": config.n_mixtures,
            "kappa_initial_bias": config.kappa_initial_bias,
        },
    }


def save_checkpoint(path, epoch, model, optimizer, dataset, records, metrics, source_checkpoint):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(epoch, model, optimizer, dataset, records, metrics, source_checkpoint),
        path,
    )


@torch.no_grad()
def evaluate_loader(model, loader, teacher_forcing_ratio):
    model.eval()
    totals = {"loss": 0.0, "mdn": 0.0, "pen": 0.0, "attn": 0.0}
    batches = 0
    for batch in loader:
        losses = batch_losses(model, batch, config.device, teacher_forcing_ratio=teacher_forcing_ratio)
        for key in totals:
            totals[key] += float(losses[key].item())
        batches += 1
    denom = max(1, batches)
    return {key: value / denom for key, value in totals.items()}


def generate_report_epoch(
    model,
    words,
    output_dir,
    epoch,
    dxdy_mean,
    dxdy_std,
    biases,
    sampling_modes,
    min_len_per_char,
    max_len_per_char,
    stop_strategy,
):
    epoch_dir = output_dir / f"epoch_{epoch:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    model.eval()
    for text in words:
        min_len = max(1, len(text) * min_len_per_char)
        max_len = max(min_len + 1, len(text) * max_len_per_char)
        for sampling_mode in sampling_modes:
            for bias in biases:
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
                    append_eos=getattr(config, "append_eos_to_generation", True),
                    eos_char=config.eos_char,
                    stop_strategy=stop_strategy,
                    sampling_mode=sampling_mode,
                    progress=False,
                    return_diagnostics=True,
                )
                stem = f"{safe_name(text)}_{sampling_mode}_bias_{str(bias).replace('.', '_')}"
                png_path = epoch_dir / f"{stem}.png"
                json_path = epoch_dir / f"{stem}.json"
                meta_path = epoch_dir / f"{stem}.meta.json"
                save_generation_plot(png_path, trajectory, f"{text} | {sampling_mode} bias={bias:g}")
                json_path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8")
                meta_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
                generated.append(
                    {
                        "text": text,
                        "sampling_mode": sampling_mode,
                        "bias": bias,
                        "png_path": str(png_path),
                        "json_path": str(json_path),
                        "meta_path": str(meta_path),
                        "diagnostics": diagnostics,
                    }
                )

    return generated


def train_epoch(model, loader, optimizer, teacher_forcing_ratio):
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        losses = batch_losses(
            model,
            batch,
            config.device,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        total_loss += float(losses["loss"].item())
        total_batches += 1

    return total_loss / max(1, total_batches)


def main():
    parser = argparse.ArgumentParser(description="Overfit the current model on a few exact words and test free-run output.")
    parser.add_argument("--checkpoints", default=config.checkpoints)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--dataset", default=config.data_path)
    parser.add_argument("--json-dir", default=str(PROJECT_ROOT / "dataset" / "jsons"))
    parser.add_argument("--text-dir", default=str(PROJECT_ROOT / "dataset" / "texts"))
    parser.add_argument("--words", nargs="*", default=None)
    parser.add_argument("--words-file", default=str(PROJECT_ROOT / "dataset" / "target_texts.txt"))
    parser.add_argument("--sample-ids", default="", help="Comma-separated trajectory ids to force into the overfit set.")
    parser.add_argument("--match", choices=["exact", "contains"], default="exact")
    parser.add_argument("--require-all-words", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.00003)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=0.98)
    parser.add_argument("--scheduled-sampling-mode", choices=["argmax", "mean", "sample"], default="mean")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs" / "overfit_words" / "latest"))
    parser.add_argument("--checkpoint-output-dir", default=str(PROJECT_ROOT / "checkpoints_overfit"))
    parser.add_argument("--biases", default="1.0")
    parser.add_argument("--sampling-modes", default="argmax")
    parser.add_argument("--min-len-per-char", type=int, default=45)
    parser.add_argument("--max-len-per-char", type=int, default=350)
    parser.add_argument("--stop-strategy", choices=["max", "mean", "min"], default="mean")
    args = parser.parse_args()

    if args.epochs < 0:
        raise ValueError("--epochs must be >= 0")

    config.scheduled_sampling_mode = args.scheduled_sampling_mode

    output_dir = Path(args.output_dir)
    checkpoint_output_dir = Path(args.checkpoint_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_output_dir.mkdir(parents=True, exist_ok=True)

    words = read_words(args.words, args.words_file)
    sample_ids = parse_csv_ints(args.sample_ids)
    biases = parse_csv_floats(args.biases) or [1.0]
    sampling_modes = parse_csv_modes(args.sampling_modes) or ["argmax"]

    records = dataset_records(args.json_dir, args.text_dir)
    selected_records, missing_words = select_records(
        records,
        words,
        sample_ids,
        args.match,
        args.require_all_words,
    )
    if not selected_records:
        raise ValueError("No overfit samples selected. Add exact word samples or pass --sample-ids.")

    dataset = HandwritingDataset(
        args.dataset,
        max_seq_len=config.max_seq_len,
        cache_prepared=False,
    )

    checkpoint_path = find_checkpoint(args.checkpoints, args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    if not checkpoint_config_matches(checkpoint):
        raise ValueError(f"Checkpoint config does not match current model: {checkpoint_path}")

    dxdy_mean = np.asarray(checkpoint.get("dxdy_mean", dataset.dxdy_mean), dtype=np.float32)
    dxdy_std = np.asarray(checkpoint.get("dxdy_std", dataset.dxdy_std), dtype=np.float32)
    dataset.dxdy_mean = dxdy_mean
    dataset.dxdy_std = dxdy_std

    subset = Subset(dataset, [record["dataset_index"] for record in selected_records])
    loader = DataLoader(
        subset,
        batch_size=min(args.batch_size, len(subset)),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=config.pin_memory,
    )
    eval_loader = DataLoader(
        subset,
        batch_size=min(args.batch_size, len(subset)),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=config.pin_memory,
    )

    model = make_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=args.lr,
        alpha=0.95,
        momentum=0.9,
        eps=1e-4,
    )

    report = {
        "source_checkpoint": str(checkpoint_path),
        "device": str(config.device),
        "epochs": args.epochs,
        "batch_size": min(args.batch_size, len(subset)),
        "learning_rate": args.lr,
        "teacher_forcing_ratio": args.teacher_forcing_ratio,
        "scheduled_sampling_mode": args.scheduled_sampling_mode,
        "requested_words": words,
        "missing_words": missing_words,
        "selected_records": selected_records,
        "history": [],
    }

    print("Source checkpoint:", checkpoint_path)
    print("Device:", config.device)
    print("Selected samples:")
    for record in selected_records:
        print(f"  trajectory_{record['sample_id']}: {record['text']}")
    if missing_words:
        print("Missing words:", ", ".join(missing_words))

    eval_every = max(1, args.eval_every)
    save_every = max(1, args.save_every)

    generation_words = list(dict.fromkeys(words + [record["text"] for record in selected_records]))
    for epoch in range(0, args.epochs + 1):
        if epoch > 0:
            train_loss = train_epoch(model, loader, optimizer, args.teacher_forcing_ratio)
        else:
            train_loss = None

        should_eval = epoch == 0 or epoch == args.epochs or epoch % eval_every == 0
        should_save = epoch > 0 and (epoch == args.epochs or epoch % save_every == 0)

        if not should_eval and not should_save:
            continue

        metrics = evaluate_loader(model, eval_loader, teacher_forcing_ratio=1.0)
        if train_loss is not None:
            metrics["train_loss"] = train_loss

        generated = []
        if should_eval:
            generated = generate_report_epoch(
                model,
                generation_words,
                output_dir,
                epoch,
                dxdy_mean,
                dxdy_std,
                biases,
                sampling_modes,
                args.min_len_per_char,
                args.max_len_per_char,
                args.stop_strategy,
            )

        if should_save:
            checkpoint_path_epoch = checkpoint_output_dir / f"epoch_{epoch}.pth"
            save_checkpoint(
                checkpoint_path_epoch,
                epoch,
                model,
                optimizer,
                dataset,
                selected_records,
                metrics,
                checkpoint_path,
            )
            save_checkpoint(
                checkpoint_output_dir / "latest.pth",
                epoch,
                model,
                optimizer,
                dataset,
                selected_records,
                metrics,
                checkpoint_path,
            )

        report["history"].append(
            {
                "epoch": epoch,
                "metrics": metrics,
                "generated": generated,
            }
        )
        print(
            f"epoch {epoch}/{args.epochs}: "
            f"loss={metrics['loss']:.4f} mdn={metrics['mdn']:.4f} "
            f"pen={metrics['pen']:.4f} attn={metrics['attn']:.4f}"
        )

    report_path = output_dir / "overfit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved report:", report_path)
    print("Saved checkpoints:", checkpoint_output_dir)


if __name__ == "__main__":
    main()
