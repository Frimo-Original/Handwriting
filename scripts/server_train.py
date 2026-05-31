import os
import re
import runpy
import sys
from pathlib import Path


sys.path.insert(0, str(Path("src").resolve()))
import config  # noqa: E402


def env(name, cast, current):
    raw = os.environ.get(name, "").strip()
    return current if raw == "" else cast(raw)


def env_bool(name, current):
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return current
    return raw in {"1", "true", "yes", "y", "on"}


def latest_epoch_in(checkpoint_dir):
    latest_epoch = 0
    for path in Path(checkpoint_dir).glob("epoch_*.pth"):
        match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
        if match:
            latest_epoch = max(latest_epoch, int(match.group(1)))
    return latest_epoch


def main():
    checkpoint_dir = Path(env("CHECKPOINTS", str, config.checkpoints)).expanduser().resolve()
    config.checkpoints = str(checkpoint_dir)
    config.batch_size = env("BATCH_SIZE", int, config.batch_size)
    config.grad_accum_steps = env("GRAD_ACCUM_STEPS", int, config.grad_accum_steps)
    config.learning_rate = env("LR", float, config.learning_rate)
    config.num_workers = env("NUM_WORKERS", int, config.num_workers)
    config.save_every = env("SAVE_EVERY", int, config.save_every)
    config.save_temporary_each_epoch = env_bool(
        "SAVE_TEMPORARY_EACH_EPOCH",
        getattr(config, "save_temporary_each_epoch", True),
    )
    config.max_seq_len = env("MAX_SEQ_LEN", int, config.max_seq_len)
    config.validation_split = env("VALIDATION_SPLIT", float, config.validation_split)
    config.eval_every = env("EVAL_EVERY", int, config.eval_every)
    config.progress_mode = env("PROGRESS_MODE", str, config.progress_mode)
    config.auto_resume = env_bool("AUTO_RESUME", config.auto_resume)
    config.pin_memory = config.device.type == "cuda"

    resume = os.environ.get("RESUME", "").strip()
    if resume:
        config.resume_checkpoint = str(Path(resume).expanduser().resolve())
        config.auto_resume = False

    final_epoch = os.environ.get("EPOCHS", "").strip()
    more_epochs = os.environ.get("MORE_EPOCHS", "").strip()
    if final_epoch:
        config.num_epochs = int(final_epoch)
    if more_epochs:
        config.num_epochs = latest_epoch_in(checkpoint_dir) + int(more_epochs)

    print("Training config:")
    print("  device:", config.device)
    print("  checkpoints:", config.checkpoints)
    print("  num_epochs:", config.num_epochs)
    print("  batch_size:", config.batch_size)
    print("  grad_accum_steps:", config.grad_accum_steps)
    print("  effective_batch:", config.batch_size * config.grad_accum_steps)
    print("  learning_rate:", config.learning_rate)
    print("  validation_split:", config.validation_split)
    print("  eval_every:", config.eval_every)
    print("  num_workers:", config.num_workers)
    print("  save_every:", config.save_every)
    print("  save_temporary_each_epoch:", config.save_temporary_each_epoch)
    print("  auto_resume:", config.auto_resume)
    print("  resume_checkpoint:", config.resume_checkpoint)

    runpy.run_path("src/run_training.py", run_name="__main__")


if __name__ == "__main__":
    main()
