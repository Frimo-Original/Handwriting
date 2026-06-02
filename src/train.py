import sys
import threading
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import config
from utils import mdn_loss


def use_live_progress():
    mode = getattr(config, "progress_mode", "compact").lower()
    if mode == "live":
        return True
    if mode == "auto":
        return sys.stderr.isatty()
    return False


def progress_log(message):
    if use_live_progress():
        tqdm.write(message)
    else:
        print(message)


@contextmanager
def heartbeat(message, every_seconds=15, enabled=True):
    if not enabled or every_seconds <= 0:
        yield
        return

    done = threading.Event()

    def worker():
        start = time.perf_counter()
        while not done.wait(every_seconds):
            elapsed = time.perf_counter() - start
            progress_log(f"{message}; still running after {elapsed / 60:.1f} min")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join(timeout=0.1)


def collate_fn(batch):
    dxy_padded = pad_sequence([b["dxy"] for b in batch], batch_first=True, padding_value=0.0)
    e_padded = pad_sequence([b["e"] for b in batch], batch_first=True, padding_value=0.0)
    text_padded = pad_sequence([b["text"] for b in batch], batch_first=True, padding_value=0)

    return {
        "dxy": dxy_padded,
        "e": e_padded,
        "text": text_padded,
        "length": torch.tensor([b["length"] for b in batch], dtype=torch.long),
        "text_lengths": torch.tensor([b["text_lengths"] for b in batch], dtype=torch.long),
    }


def pen_up_pos_weight(e_target):
    weight = getattr(config, "pen_up_pos_weight", None)
    if weight in (None, False, 0):
        return None
    if isinstance(weight, str) and weight.lower() == "auto":
        positives = e_target.sum().clamp_min(1.0)
        negatives = e_target.numel() - positives
        max_weight = getattr(config, "max_pen_up_pos_weight", 100.0)
        return (negatives / positives).clamp(min=1.0, max=max_weight).detach().reshape(1)
    return torch.tensor([float(weight)], device=e_target.device, dtype=e_target.dtype)


def kappa_progress_loss(kappa, lengths, text_lengths):
    B, T, _ = kappa.shape
    device = kappa.device
    dtype = kappa.dtype

    mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    time = torch.arange(1, T + 1, device=device, dtype=dtype).unsqueeze(0)
    target = time / lengths.to(dtype).clamp_min(1).unsqueeze(1)
    target = target * text_lengths.to(dtype).clamp_min(1).unsqueeze(1)
    pred = kappa.mean(dim=2)

    return F.smooth_l1_loss(pred[mask], target[mask])


def batch_losses(model, batch, device, teacher_forcing_ratio=1.0):
    dxy = batch["dxy"].to(device, non_blocking=config.pin_memory)
    e_target = batch["e"].to(device, non_blocking=config.pin_memory)
    text = batch["text"].to(device, non_blocking=config.pin_memory)
    text_lengths = batch["text_lengths"].to(device, non_blocking=config.pin_memory)
    lengths = batch["length"].to(device, non_blocking=config.pin_memory)

    outputs = model(
        dxy,
        e_target,
        text,
        text_lengths,
        teacher_forcing_ratio=teacher_forcing_ratio,
        scheduled_sampling_mode=getattr(config, "scheduled_sampling_mode", "argmax"),
    )

    B, T = dxy.shape[:2]
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
    attention_weight = getattr(config, "attention_loss_weight", 0.0)
    loss = loss_mdn + loss_e + attention_weight * loss_attn
    return {
        "loss": loss,
        "mdn": loss_mdn.detach(),
        "pen": loss_e.detach(),
        "attn": loss_attn.detach(),
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    totals = {"loss": 0.0, "mdn": 0.0, "pen": 0.0, "attn": 0.0}
    batches = 0

    for batch in dataloader:
        losses = batch_losses(model, batch, device)
        for key in totals:
            value = losses[key]
            totals[key] += float(value.item() if torch.is_tensor(value) else value)
        batches += 1

    denom = max(batches, 1)
    return {key: value / denom for key, value in totals.items()}


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_batches = 0
    grad_accum_steps = getattr(config, "grad_accum_steps", 1)
    teacher_forcing_ratio = float(getattr(config, "teacher_forcing_ratio", 1.0))
    compiled_model = getattr(model, "_orig_mod", None) is not None
    compile_heartbeat_seconds = getattr(config, "compile_heartbeat_seconds", 15)

    live_progress = use_live_progress()
    progress = (
        tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}",
            unit="batch",
            leave=False,
            dynamic_ncols=False,
            ncols=getattr(config, "progress_ncols", 120),
            mininterval=getattr(config, "progress_mininterval", 2.0),
            ascii=config.tqdm_ascii,
            file=sys.stdout,
            position=0,
        )
        if live_progress
        else dataloader
    )

    for batch_idx, batch in enumerate(progress):
        first_compiled_batch = compiled_model and batch_idx == 0
        if first_compiled_batch:
            progress_log(
                "torch.compile: first batch starts now. "
                "PyTorch may spend several minutes compiling before batch progress moves."
            )
            if live_progress:
                progress.set_description(f"Epoch {epoch + 1} compile forward")
                progress.refresh()

        forward_started = time.perf_counter()
        with heartbeat(
            f"torch.compile forward epoch={epoch + 1} batch=1",
            every_seconds=compile_heartbeat_seconds,
            enabled=first_compiled_batch,
        ):
            losses = batch_losses(
                model,
                batch,
                device,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
        forward_seconds = time.perf_counter() - forward_started
        loss = losses["loss"]
        loss_mdn = losses["mdn"]
        loss_e = losses["pen"]
        loss_attn = losses["attn"]

        if first_compiled_batch and live_progress:
            progress.set_description(f"Epoch {epoch + 1} compile backward")
            progress.refresh()

        backward_started = time.perf_counter()
        with heartbeat(
            f"torch.compile backward epoch={epoch + 1} batch=1",
            every_seconds=compile_heartbeat_seconds,
            enabled=first_compiled_batch,
        ):
            (loss / grad_accum_steps).backward()
        backward_seconds = time.perf_counter() - backward_started

        should_step = (batch_idx + 1) % grad_accum_steps == 0 or batch_idx + 1 == len(dataloader)
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        total_batches += 1
        avg_loss = total_loss / total_batches

        postfix = {
            "loss": f"{avg_loss:.4f}",
            "mdn": f"{loss_mdn.item():.4f}",
            "pen": f"{loss_e.item():.4f}",
            "attn": f"{loss_attn.item():.4f}",
            "tf": f"{teacher_forcing_ratio:.2f}",
        }
        if device.type == "cuda":
            postfix["gpu"] = f"{torch.cuda.memory_allocated(device) / 1024**3:.2f}GB"
        if live_progress:
            progress.set_postfix(postfix)

        if first_compiled_batch:
            B, T = batch["dxy"].shape[:2]
            progress_log(
                "torch.compile first batch finished: "
                f"B={B}, T={T}, forward={forward_seconds:.1f}s, backward={backward_seconds:.1f}s"
            )
            if live_progress:
                progress.set_description(f"Epoch {epoch + 1}")

    return total_loss / max(total_batches, 1)
