# -*- coding: utf-8 -*-
import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from dataset import HandwritingDataset
from model import HandwritingSynthesis
from train import collate_fn, kappa_progress_loss, pen_up_pos_weight
from utils import mdn_loss


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed(device, fn):
    sync(device)
    start = time.perf_counter()
    result = fn()
    sync(device)
    return time.perf_counter() - start, result


def find_latest_checkpoint(checkpoint_dir):
    paths = []
    for path in Path(checkpoint_dir).glob("epoch_*.pth"):
        match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
        if match:
            paths.append((int(match.group(1)), path))
    return max(paths, key=lambda item: item[0])[1] if paths else None


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


def transfer_batch(batch, device):
    return {
        "dxy": batch["dxy"].to(device, non_blocking=config.pin_memory),
        "e": batch["e"].to(device, non_blocking=config.pin_memory),
        "text": batch["text"].to(device, non_blocking=config.pin_memory),
        "length": batch["length"].to(device, non_blocking=config.pin_memory),
        "text_lengths": batch["text_lengths"].to(device, non_blocking=config.pin_memory),
    }


def compute_loss(outputs, batch):
    dxy = batch["dxy"]
    e_target = batch["e"]
    lengths = batch["length"]
    text_lengths = batch["text_lengths"]

    B, T = dxy.shape[:2]
    device = dxy.device
    mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    flat_mask = mask.reshape(-1)

    loss_mdn = mdn_loss(
        outputs["pi"].reshape(B * T, -1)[flat_mask],
        outputs["mu"].reshape(B * T, outputs["mu"].shape[2], 2)[flat_mask],
        outputs["sigma"].reshape(B * T, outputs["sigma"].shape[2], 2)[flat_mask],
        outputs["rho"].reshape(B * T, -1)[flat_mask],
        dxy.reshape(B * T, 2)[flat_mask],
        reduction="mean",
    )
    e_logits = outputs["e_logit"].reshape(B * T, 1)[flat_mask]
    e_targets = e_target.reshape(B * T, 1)[flat_mask]
    loss_e = F.binary_cross_entropy_with_logits(
        e_logits,
        e_targets,
        pos_weight=pen_up_pos_weight(e_targets),
    )
    loss_attn = kappa_progress_loss(outputs["kappa"], lengths, text_lengths)
    loss = loss_mdn + loss_e + config.attention_loss_weight * loss_attn
    return loss, {"mdn": loss_mdn, "pen": loss_e, "attn": loss_attn}


def print_table(totals, measured_batches):
    total_time = sum(totals.values())
    print("\nTiming summary")
    print(f"{'stage':<16} {'total_s':>10} {'avg_ms':>10} {'pct':>8}")
    print("-" * 48)
    for key, value in sorted(totals.items(), key=lambda item: item[1], reverse=True):
        pct = 100.0 * value / total_time if total_time > 0 else 0.0
        avg_ms = 1000.0 * value / max(measured_batches, 1)
        print(f"{key:<16} {value:10.3f} {avg_ms:10.1f} {pct:7.1f}%")
    print("-" * 48)
    print(f"{'total':<16} {total_time:10.3f} {1000.0 * total_time / max(measured_batches, 1):10.1f} {100.0:7.1f}%")


def print_hints(totals):
    if not totals:
        return
    dominant = max(totals.items(), key=lambda item: item[1])[0]
    print("\nHow to read this")
    if dominant == "data_wait":
        print("- data_wait dominates: DataLoader/collate/CPU preparation is the bottleneck.")
        print("  Try num_workers=2/4, cache_prepared_dataset=True, and check disk/CPU load.")
    elif dominant == "forward":
        print("- forward dominates: model sequence computation is the bottleneck.")
        print("  Try lowering max_seq_len, reducing lstm_size/K/n_mixtures, or experimenting with torch.compile.")
    elif dominant == "backward":
        print("- backward dominates: gradients through the long recurrent graph are the bottleneck.")
        print("  Try lower max_seq_len, smaller model, or larger batch only if GPU is underutilized.")
    elif dominant == "loss":
        print("- loss dominates: MDN/attention loss masking and tensor reshaping are costly.")
        print("  This points to optimizing loss computation or reducing T/max_seq_len.")
    elif dominant == "step":
        print("- step dominates: optimizer/gradient clipping is unexpectedly costly.")
        print("  Check grad_clip cost and optimizer state size.")
    elif dominant == "transfer":
        print("- transfer dominates: CPU->GPU copy is costly.")
        print("  Try pin_memory=True, fewer tiny batches, and larger batch_size if memory allows.")


def main():
    parser = argparse.ArgumentParser(description="Profile training bottlenecks on a few batches.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoints", default=config.checkpoints)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--max-seq-len", type=int, default=config.max_seq_len)
    parser.add_argument("--num-workers", type=int, default=config.num_workers)
    parser.add_argument("--no-load-checkpoint", action="store_true")
    args = parser.parse_args()

    config.batch_size = args.batch_size
    config.max_seq_len = args.max_seq_len
    config.num_workers = args.num_workers
    config.pin_memory = config.device.type == "cuda"

    if config.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(config.device)
        torch.backends.cudnn.benchmark = config.cudnn_benchmark

    dataset = HandwritingDataset(
        config.data_path,
        max_seq_len=config.max_seq_len,
        cache_prepared=getattr(config, "cache_prepared_dataset", True),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    model = make_model()
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint(args.checkpoints)
    if checkpoint_path and not args.no_load_checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=config.learning_rate,
        alpha=0.95,
        momentum=0.9,
        eps=1e-4,
    )
    model.train()

    print("Training profiler")
    print("device:", config.device)
    if config.device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(config.device))
    print("checkpoint:", checkpoint_path if checkpoint_path and not args.no_load_checkpoint else "not loaded")
    print("samples:", len(dataset))
    print("batch_size:", config.batch_size)
    print("max_seq_len:", config.max_seq_len)
    print("num_workers:", config.num_workers)
    print("pin_memory:", config.pin_memory)
    print("warmup_batches:", args.warmup)
    print("measured_batches:", args.batches)

    iterator = iter(dataloader)
    totals = defaultdict(float)
    measured = 0
    last_batch_shape = None

    for batch_idx in range(args.warmup + args.batches):
        data_seconds, batch = elapsed(config.device, lambda: next(iterator))
        transferred_seconds, batch_gpu = elapsed(config.device, lambda: transfer_batch(batch, config.device))
        forward_seconds, outputs = elapsed(
            config.device,
            lambda: model(batch_gpu["dxy"], batch_gpu["e"], batch_gpu["text"], batch_gpu["text_lengths"]),
        )
        loss_seconds, loss_pack = elapsed(config.device, lambda: compute_loss(outputs, batch_gpu))
        loss, parts = loss_pack
        backward_seconds, _ = elapsed(config.device, lambda: loss.backward())

        def optimizer_step():
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        step_seconds, _ = elapsed(config.device, optimizer_step)

        if batch_idx >= args.warmup:
            totals["data_wait"] += data_seconds
            totals["transfer"] += transferred_seconds
            totals["forward"] += forward_seconds
            totals["loss"] += loss_seconds
            totals["backward"] += backward_seconds
            totals["step"] += step_seconds
            measured += 1
            B, T = batch["dxy"].shape[:2]
            last_batch_shape = (B, T)
            print(
                f"batch {measured:02d}: B={B} T={T} "
                f"data={data_seconds:.3f}s transfer={transferred_seconds:.3f}s "
                f"forward={forward_seconds:.3f}s loss={loss_seconds:.3f}s "
                f"backward={backward_seconds:.3f}s step={step_seconds:.3f}s "
                f"loss={loss.item():.4f}"
            )

    print_table(totals, measured)
    if last_batch_shape:
        print("last_batch_shape:", {"B": last_batch_shape[0], "T": last_batch_shape[1]})
    if config.device.type == "cuda":
        print("max_cuda_memory_gb:", round(torch.cuda.max_memory_allocated(config.device) / 1024**3, 3))
    print_hints(totals)


if __name__ == "__main__":
    main()
