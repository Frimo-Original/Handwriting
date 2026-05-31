import os
import re
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import config
from dataset import HandwritingDataset
from model import HandwritingSynthesis
from train import collate_fn, evaluate, train_one_epoch, use_live_progress


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
    val_fraction = float(getattr(config, "validation_split", 0.0) or 0.0)
    if len(dataset) >= 2 and val_fraction > 0:
        val_size = max(1, int(round(len(dataset) * val_fraction)))
        val_size = min(val_size, len(dataset) - 1)
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(getattr(config, "validation_seed", 20260531))
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
    return dataset, train_loader, val_loader


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


def checkpoint_payload(epoch, model, optimizer, dataset, metrics, best_val_loss, checkpoint_type):
    return {
        "epoch": epoch,
        "model_state_dict": unwrap_compiled_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "dxdy_mean": dataset.dxdy_mean,
        "dxdy_std": dataset.dxdy_std,
        "best_val_loss": best_val_loss,
        "metrics": metrics,
        "checkpoint_type": checkpoint_type,
        "temporary": checkpoint_type == "temporary",
        "config": {
            "vocab_size": config.vocab_size,
            "embed_dim": config.embed_dim,
            "lstm_size": config.lstm_size,
            "K": config.K,
            "n_mixtures": config.n_mixtures,
            "kappa_initial_bias": config.kappa_initial_bias,
        },
    }


def save_checkpoint(path, epoch, model, optimizer, dataset, metrics, best_val_loss, checkpoint_type):
    torch.save(
        checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            metrics=metrics,
            best_val_loss=best_val_loss,
            checkpoint_type=checkpoint_type,
        ),
        path,
    )


def is_permanent_epoch(epoch):
    save_every = max(1, int(getattr(config, "save_every", 1)))
    return epoch % save_every == 0 or epoch == config.num_epochs


def maybe_delete_temporary_checkpoint(path):
    if not path:
        return
    path = Path(path)
    if not path.exists():
        return
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        log(f"Skip deleting previous temporary checkpoint {path}: cannot read it ({exc})")
        return
    if checkpoint.get("temporary") or checkpoint.get("checkpoint_type") == "temporary":
        path.unlink()
        log(f"Deleted previous temporary checkpoint: {path}")


stage("Preparing dataset and DataLoader...")
dataset, dataloader, val_dataloader = make_dataloader()
dataset_mtime = os.path.getmtime(DATA_PATH)
val_count = len(val_dataloader.dataset) if val_dataloader is not None else 0
stage(f"Loaded dataset: {len(dataset)} samples; train={len(dataloader.dataset)}, val={val_count}")

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
best_val_loss = float("inf")
previous_temporary_checkpoint = None
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
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        if checkpoint.get("temporary") or checkpoint.get("checkpoint_type") == "temporary":
            previous_temporary_checkpoint = resume_path
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
    metrics = {"train_loss": loss}

    should_eval = (
        val_dataloader is not None
        and getattr(config, "eval_every", 1)
        and ((epoch + 1) % int(config.eval_every) == 0 or epoch + 1 == config.num_epochs)
    )
    if should_eval:
        val_metrics = evaluate(model, val_dataloader, config.device)
        metrics.update({f"val_{key}": value for key, value in val_metrics.items()})

    metric_text = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    log(f"Epoch {epoch + 1}/{config.num_epochs}: {metric_text}")

    epoch_number = epoch + 1
    best_improved = False
    if val_dataloader is not None and getattr(config, "save_best", True):
        val_loss = metrics.get("val_loss")
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_improved = True

    checkpoint_is_permanent = is_permanent_epoch(epoch_number)
    should_save = checkpoint_is_permanent or getattr(config, "save_temporary_each_epoch", True)
    if should_save:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch + 1}.pth")
        checkpoint_type = "permanent" if checkpoint_is_permanent else "temporary"
        save_checkpoint(
            checkpoint_path,
            epoch_number,
            model,
            optimizer,
            dataset,
            metrics,
            best_val_loss,
            checkpoint_type,
        )
        log(f"Saved {checkpoint_type} checkpoint: {checkpoint_path}")

        if previous_temporary_checkpoint and str(previous_temporary_checkpoint) != str(checkpoint_path):
            maybe_delete_temporary_checkpoint(previous_temporary_checkpoint)
        previous_temporary_checkpoint = None if checkpoint_is_permanent else checkpoint_path

    if best_improved:
        best_path = os.path.join(CHECKPOINT_DIR, "best.pth")
        save_checkpoint(
            best_path,
            epoch_number,
            model,
            optimizer,
            dataset,
            metrics,
            best_val_loss,
            "best",
        )
        log(f"Saved best checkpoint: {best_path} val_loss={best_val_loss:.4f}")

    if config.device.type == "cuda" and config.empty_cache_each_epoch:
        torch.cuda.empty_cache()

    if config.reload_dataset_each_epoch:
        current_mtime = os.path.getmtime(DATA_PATH)
        if current_mtime != dataset_mtime:
            dataset, dataloader, val_dataloader = make_dataloader()
            dataset_mtime = current_mtime
            val_count = len(val_dataloader.dataset) if val_dataloader is not None else 0
            log(f"Reloaded dataset: {len(dataset)} samples; train={len(dataloader.dataset)}, val={val_count}")

print("Training finished.")
