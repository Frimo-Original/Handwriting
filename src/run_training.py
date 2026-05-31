import os
import re
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import HandwritingDataset
from model import HandwritingSynthesis
from train import collate_fn, train_one_epoch, use_live_progress


DATA_PATH = config.data_path
CHECKPOINT_DIR = config.checkpoints
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
LIVE_PROGRESS = use_live_progress()
log = tqdm.write if LIVE_PROGRESS else print
RUN_STARTED_AT = time.perf_counter()


def stage(message):
    elapsed = time.perf_counter() - RUN_STARTED_AT
    log(f"[startup {elapsed:6.1f}s] {message}")

if config.device.type == "cuda":
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.cuda.set_device(0)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("Using CPU")

def make_dataloader():
    dataset = HandwritingDataset(
        DATA_PATH,
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
    return dataset, dataloader


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_paths = []
    for path in Path(checkpoint_dir).glob("epoch_*.pth"):
        match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
        if match:
            checkpoint_paths.append((int(match.group(1)), path))
    if not checkpoint_paths:
        return None
    return max(checkpoint_paths, key=lambda item: item[0])[1]


def checkpoint_config_matches(checkpoint):
    saved = checkpoint.get("config", {})
    expected = {
        "vocab_size": config.vocab_size,
        "embed_dim": config.embed_dim,
        "lstm_size": config.lstm_size,
        "K": config.K,
        "n_mixtures": config.n_mixtures,
        "kappa_initial_bias": config.kappa_initial_bias,
    }
    return all(saved.get(key) == value for key, value in expected.items())


def maybe_compile_model(model):
    if not getattr(config, "compile_model", False):
        return model
    if not hasattr(torch, "compile"):
        log("torch.compile is not available in this PyTorch version; using eager model.")
        return model

    try:
        compiled = torch.compile(model, mode=getattr(config, "compile_mode", "default"))
    except Exception as exc:
        log(f"torch.compile failed during setup; using eager model. Reason: {exc}")
        return model

    log(f"Using torch.compile mode={getattr(config, 'compile_mode', 'default')}")
    return compiled


def unwrap_compiled_model(model):
    return getattr(model, "_orig_mod", model)


stage("Preparing dataset and DataLoader...")
dataset, dataloader = make_dataloader()
dataset_mtime = os.path.getmtime(DATA_PATH)
stage(f"Loaded dataset: {len(dataset)} samples")

stage("Creating model...")
model = HandwritingSynthesis(
    vocab_size=config.vocab_size,
    embed_dim=config.embed_dim,
    lstm_size=config.lstm_size,
    num_layers=config.num_lstm_layers,
    K=config.K,
    n_mixtures=config.n_mixtures,
    kappa_initial_bias=config.kappa_initial_bias,
).to(config.device)
stage("Model created")

stage("Creating optimizer...")
optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=config.learning_rate,
    alpha=0.95,
    momentum=0.9,
    eps=1e-4,
)
stage("Optimizer created")

start_epoch = 0
resume_path = config.resume_checkpoint
if resume_path is None and config.auto_resume:
    latest_checkpoint = find_latest_checkpoint(CHECKPOINT_DIR)
    resume_path = str(latest_checkpoint) if latest_checkpoint else None

if resume_path:
    stage(f"Loading checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=config.device, weights_only=False)
    if not checkpoint_config_matches(checkpoint):
        log(f"Skip resume: checkpoint config does not match current model: {resume_path}")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(config.device)
        else:
            log("Checkpoint has no optimizer state; optimizer will start fresh.")
        start_epoch = int(checkpoint.get("epoch", 0))
        if start_epoch == 0:
            match = re.search(r"epoch_(\d+)\.pth$", str(resume_path))
            if match:
                start_epoch = int(match.group(1))
        stage(f"Resumed from {resume_path}; next epoch: {start_epoch + 1}")

stage("Preparing torch.compile wrapper...")
model = maybe_compile_model(model)
stage("Training loop starts")

for epoch in range(start_epoch, config.num_epochs):
    loss = train_one_epoch(model, dataloader, optimizer, config.device, epoch)
    log(f"Epoch {epoch + 1}/{config.num_epochs}: loss={loss:.4f}")

    should_save = (epoch + 1) % config.save_every == 0 or epoch + 1 == config.num_epochs
    if should_save:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch + 1}.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": unwrap_compiled_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dxdy_mean": dataset.dxdy_mean,
                "dxdy_std": dataset.dxdy_std,
                "config": {
                    "vocab_size": config.vocab_size,
                    "embed_dim": config.embed_dim,
                    "lstm_size": config.lstm_size,
                    "K": config.K,
                    "n_mixtures": config.n_mixtures,
                    "kappa_initial_bias": config.kappa_initial_bias,
                },
            },
            checkpoint_path,
        )
        log(f"Saved checkpoint: {checkpoint_path}")

    if config.device.type == "cuda" and config.empty_cache_each_epoch:
        torch.cuda.empty_cache()

    if config.reload_dataset_each_epoch:
        current_mtime = os.path.getmtime(DATA_PATH)
        if current_mtime != dataset_mtime:
            dataset, dataloader = make_dataloader()
            dataset_mtime = current_mtime
            log(f"Reloaded dataset: {len(dataset)} samples")

print("Training finished.")
