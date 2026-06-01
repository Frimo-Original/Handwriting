# -*- coding: utf-8 -*-
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import config  # noqa: E402
from dataset import HandwritingDataset  # noqa: E402
from train import collate_fn  # noqa: E402


def pick_indices(lengths, batch_size, mode):
    if mode == "first":
        return list(range(min(batch_size, len(lengths))))

    ordered = sorted(range(len(lengths)), key=lambda idx: lengths[idx])
    if mode == "longest":
        return list(reversed(ordered[-batch_size:]))

    if mode == "shortest":
        return ordered[:batch_size]

    start = max(0, (len(ordered) - batch_size) // 2)
    return ordered[start : start + batch_size]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare one PyTorch batch for the C++ train-step benchmark.")
    parser.add_argument("--output", default=str(ROOT / "cpp_experiments" / "batches" / "batch.bin"))
    parser.add_argument("--batch-size", type=int, default=getattr(config, "batch_size", 24))
    parser.add_argument("--max-seq-len", type=int, default=getattr(config, "max_seq_len", 3000))
    parser.add_argument(
        "--mode",
        choices=["first", "median", "longest", "shortest"],
        default="median",
        help="Which samples to pack into the benchmark batch.",
    )
    return parser.parse_args()


def write_tensor(handle, tensor):
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.float32:
        dtype_code = 0
        array = tensor.numpy().astype(np.float32, copy=False)
    elif tensor.dtype == torch.int64:
        dtype_code = 1
        array = tensor.numpy().astype(np.int64, copy=False)
    else:
        raise TypeError(f"Unsupported tensor dtype: {tensor.dtype}")

    handle.write(struct.pack("<ii", dtype_code, tensor.dim()))
    for dim in tensor.shape:
        handle.write(struct.pack("<q", int(dim)))
    handle.write(array.tobytes(order="C"))


def save_batch(path, tensors):
    with path.open("wb") as handle:
        handle.write(b"HWBATCH1\n")
        handle.write(struct.pack("<i", len(tensors)))
        for tensor in tensors:
            write_tensor(handle, tensor)


def main():
    args = parse_args()
    dataset = HandwritingDataset(
        config.data_path,
        max_seq_len=args.max_seq_len,
        cache_prepared=True,
    )
    lengths = [min(int(len(points)), args.max_seq_len) for points in dataset.points]
    indices = pick_indices(lengths, args.batch_size, args.mode)
    batch = collate_fn([dataset[idx] for idx in indices])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_batch(
        output,
        [
            batch["dxy"].contiguous(),
            batch["e"].contiguous(),
            batch["text"].contiguous(),
            batch["text_lengths"].contiguous(),
            batch["length"].contiguous(),
        ],
    )

    metadata = {
        "batch_size": int(batch["dxy"].shape[0]),
        "max_T": int(batch["dxy"].shape[1]),
        "max_text_length": int(batch["text"].shape[1]),
        "indices": indices,
        "lengths": [int(lengths[idx]) for idx in indices],
        "text_lengths": [int(value) for value in batch["text_lengths"].tolist()],
        "mode": args.mode,
        "vocab_size": int(config.vocab_size),
        "eos_token": config.eos_token,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved batch:", output)
    print("Metadata:", output.with_suffix(".json"))
    print("Shape dxy:", tuple(batch["dxy"].shape))
    print("Shape text:", tuple(batch["text"].shape))


if __name__ == "__main__":
    main()
