# -*- coding: utf-8 -*-
import argparse
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

import config
from dataset import HandwritingDataset
from model import HandwritingSynthesis
from train import collate_fn, evaluate


def find_checkpoint(checkpoint_dir, checkpoint):
    if checkpoint:
        return Path(checkpoint).expanduser().resolve()

    best = Path(checkpoint_dir) / "best.pth"
    if best.exists():
        return best

    paths = []
    for path in Path(checkpoint_dir).glob("epoch_*.pth"):
        match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
        if match:
            paths.append((int(match.group(1)), path))
    if not paths:
        raise FileNotFoundError(f"No best.pth or epoch_*.pth found in {checkpoint_dir}")
    return max(paths, key=lambda item: item[0])[1]


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


def make_validation_loader():
    dataset = HandwritingDataset(
        config.data_path,
        max_seq_len=config.max_seq_len,
        cache_prepared=getattr(config, "cache_prepared_dataset", True),
    )
    val_fraction = float(getattr(config, "validation_split", 0.0) or 0.0)
    if len(dataset) < 2 or val_fraction <= 0:
        val_dataset = dataset
    else:
        val_size = max(1, int(round(len(dataset) * val_fraction)))
        val_size = min(val_size, len(dataset) - 1)
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(getattr(config, "validation_seed", 20260531))
        _, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    return DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a handwriting checkpoint on the configured validation split.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoints", default=config.checkpoints)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    checkpoint_path = find_checkpoint(args.checkpoints, args.checkpoint)
    model = make_model()
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = make_validation_loader()
    metrics = evaluate(model, loader, config.device)

    result = {
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "samples": len(loader.dataset),
        "metrics": metrics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
