SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.RECIPEPREFIX := >
.ONESHELL:
.DEFAULT_GOAL := help

PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cu121

CHECKPOINTS ?= checkpoints_attention
EPOCHS ?=
MORE_EPOCHS ?=
BATCH_SIZE ?=
GRAD_ACCUM_STEPS ?=
LR ?=
NUM_WORKERS ?= 2
SAVE_EVERY ?=
MAX_SEQ_LEN ?=
PROGRESS_MODE ?= live
AUTO_RESUME ?= true
RESUME ?=
LOG ?= training.log

TEXT ?= sample
BIAS ?= 1.0
MIN_GEN_LEN ?= 200
MAX_GEN_LEN ?= 3000
OUTPUT_JSON ?= generated_trajectory.json
OUTPUT_PNG ?= generated_trajectory.png

ifneq ($(strip $(CUDA_VISIBLE_DEVICES)),)
CUDA_PREFIX := CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)
else
CUDA_PREFIX :=
endif

.PHONY: help venv check dataset train train-log latest generate clean-npzs

help:
> @printf '%s\n' \
>   'Targets:' \
>   '  make venv                         Create .venv and install dependencies' \
>   '  make check                        Show Python, PyTorch, CUDA and dataset status' \
>   '  make dataset                      Rebuild dataset/all_trajectories.npz' \
>   '  make train                        Continue training from latest checkpoint' \
>   '  make train-log                    Continue training and append output to LOG' \
>   '  make latest                       Print latest checkpoint path' \
>   '  make generate TEXT="..."          Generate JSON/PNG from latest checkpoint' \
>   '' \
>   'Useful variables:' \
>   '  MORE_EPOCHS=20                    Train 20 epochs after the latest checkpoint' \
>   '  EPOCHS=500                        Train until absolute epoch 500' \
>   '  BATCH_SIZE=8 GRAD_ACCUM_STEPS=1   Override training batch settings' \
>   '  NUM_WORKERS=4                     DataLoader workers on Linux' \
>   '  CHECKPOINTS=checkpoints_attention Checkpoint directory' \
>   '  CUDA_VISIBLE_DEVICES=0            Select GPU visible to PyTorch' \
>   '  TORCH_INDEX_URL=.../cu118         Use another PyTorch CUDA wheel index if needed' \
>   '' \
>   'Examples:' \
>   '  make venv check dataset' \
>   '  make train MORE_EPOCHS=10 BATCH_SIZE=8 GRAD_ACCUM_STEPS=1' \
>   '  nohup make train-log MORE_EPOCHS=100 LOG=v100.log &' \
>   '  make generate TEXT="mama" BIAS=1.0'

$(PY):
> $(PYTHON) -m venv "$(VENV)"

$(VENV)/.deps-installed: $(PY)
> "$(PIP)" install --upgrade pip
> "$(PIP)" install numpy tqdm matplotlib
> "$(PIP)" install torch --index-url "$(TORCH_INDEX_URL)"
> touch "$@"

venv: $(VENV)/.deps-installed

check: venv
> nvidia-smi || true
> "$(PY)" - <<'PY'
> import sys
> from pathlib import Path
> import torch
>
> print("Python:", sys.version.replace("\n", " "))
> print("PyTorch:", torch.__version__)
> print("CUDA available:", torch.cuda.is_available())
> if torch.cuda.is_available():
>     print("GPU count:", torch.cuda.device_count())
>     for idx in range(torch.cuda.device_count()):
>         props = torch.cuda.get_device_properties(idx)
>         print(f"GPU {idx}: {torch.cuda.get_device_name(idx)} ({props.total_memory / 1024**3:.2f} GB)")
>
> paths = [
>     Path("src/config.py"),
>     Path("src/run_training.py"),
>     Path("dataset/jsons"),
>     Path("dataset/texts"),
>     Path("dataset/all_trajectories.npz"),
> ]
> for path in paths:
>     status = "ok" if path.exists() else "missing"
>     print(f"{status:7} {path}")
> print("json files:", len(list(Path("dataset/jsons").glob("trajectory_*.json"))))
> print("txt files: ", len(list(Path("dataset/texts").glob("trajectory_*.txt"))))
> PY

dataset: venv
> mkdir -p dataset/npzs
> "$(PY)" dataset/converter.py

latest:
> if [ -d "$(CHECKPOINTS)" ]; then find "$(CHECKPOINTS)" -maxdepth 1 -name 'epoch_*.pth' | sort -V | tail -n 1; else echo "Missing checkpoint dir: $(CHECKPOINTS)"; fi

train: dataset
> $(CUDA_PREFIX) CHECKPOINTS="$(CHECKPOINTS)" EPOCHS="$(EPOCHS)" MORE_EPOCHS="$(MORE_EPOCHS)" BATCH_SIZE="$(BATCH_SIZE)" GRAD_ACCUM_STEPS="$(GRAD_ACCUM_STEPS)" LR="$(LR)" NUM_WORKERS="$(NUM_WORKERS)" SAVE_EVERY="$(SAVE_EVERY)" MAX_SEQ_LEN="$(MAX_SEQ_LEN)" PROGRESS_MODE="$(PROGRESS_MODE)" AUTO_RESUME="$(AUTO_RESUME)" RESUME="$(RESUME)" "$(PY)" - <<'PY'
> import os
> import re
> import runpy
> import sys
> from pathlib import Path
>
> sys.path.insert(0, str(Path("src").resolve()))
> import config
>
> def env(name, cast, current):
>     raw = os.environ.get(name, "").strip()
>     return current if raw == "" else cast(raw)
>
> def env_bool(name, current):
>     raw = os.environ.get(name, "").strip().lower()
>     if raw == "":
>         return current
>     return raw in {"1", "true", "yes", "y", "on"}
>
> checkpoint_dir = Path(env("CHECKPOINTS", str, config.checkpoints)).expanduser().resolve()
> config.checkpoints = str(checkpoint_dir)
> config.batch_size = env("BATCH_SIZE", int, config.batch_size)
> config.grad_accum_steps = env("GRAD_ACCUM_STEPS", int, config.grad_accum_steps)
> config.learning_rate = env("LR", float, config.learning_rate)
> config.num_workers = env("NUM_WORKERS", int, config.num_workers)
> config.save_every = env("SAVE_EVERY", int, config.save_every)
> config.max_seq_len = env("MAX_SEQ_LEN", int, config.max_seq_len)
> config.progress_mode = env("PROGRESS_MODE", str, config.progress_mode)
> config.auto_resume = env_bool("AUTO_RESUME", config.auto_resume)
> config.pin_memory = config.device.type == "cuda"
>
> resume = os.environ.get("RESUME", "").strip()
> if resume:
>     config.resume_checkpoint = str(Path(resume).expanduser().resolve())
>     config.auto_resume = False
>
> final_epoch = os.environ.get("EPOCHS", "").strip()
> more_epochs = os.environ.get("MORE_EPOCHS", "").strip()
> if final_epoch:
>     config.num_epochs = int(final_epoch)
> if more_epochs:
>     latest_epoch = 0
>     for path in checkpoint_dir.glob("epoch_*.pth"):
>         match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
>         if match:
>             latest_epoch = max(latest_epoch, int(match.group(1)))
>     config.num_epochs = latest_epoch + int(more_epochs)
>
> print("Training config:")
> print("  device:", config.device)
> print("  checkpoints:", config.checkpoints)
> print("  num_epochs:", config.num_epochs)
> print("  batch_size:", config.batch_size)
> print("  grad_accum_steps:", config.grad_accum_steps)
> print("  effective_batch:", config.batch_size * config.grad_accum_steps)
> print("  learning_rate:", config.learning_rate)
> print("  num_workers:", config.num_workers)
> print("  auto_resume:", config.auto_resume)
> print("  resume_checkpoint:", config.resume_checkpoint)
>
> runpy.run_path("src/run_training.py", run_name="__main__")
> PY

train-log:
> $(MAKE) train 2>&1 | tee -a "$(LOG)"

generate: venv
> $(CUDA_PREFIX) CHECKPOINTS="$(CHECKPOINTS)" TEXT="$(TEXT)" BIAS="$(BIAS)" MIN_GEN_LEN="$(MIN_GEN_LEN)" MAX_GEN_LEN="$(MAX_GEN_LEN)" OUTPUT_JSON="$(OUTPUT_JSON)" OUTPUT_PNG="$(OUTPUT_PNG)" "$(PY)" - <<'PY'
> import json
> import os
> import re
> import sys
> from pathlib import Path
>
> import matplotlib
> matplotlib.use("Agg")
> import matplotlib.pyplot as plt
> import numpy as np
> import torch
>
> sys.path.insert(0, str(Path("src").resolve()))
> import config
> from generate import generate
> from model import HandwritingSynthesis
>
> def latest_checkpoint(checkpoint_dir):
>     paths = []
>     for path in Path(checkpoint_dir).glob("epoch_*.pth"):
>         match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
>         if match:
>             paths.append((int(match.group(1)), path))
>     if not paths:
>         raise FileNotFoundError(f"No epoch_*.pth found in {checkpoint_dir}")
>     return max(paths, key=lambda item: item[0])[1]
>
> config.checkpoints = str(Path(os.environ.get("CHECKPOINTS", config.checkpoints)).expanduser().resolve())
> text = os.environ.get("TEXT", "sample")
> bias = float(os.environ.get("BIAS", "1.0"))
> min_len = int(os.environ.get("MIN_GEN_LEN", "200"))
> max_len = int(os.environ.get("MAX_GEN_LEN", "3000"))
> output_json = os.environ.get("OUTPUT_JSON", "generated_trajectory.json")
> output_png = os.environ.get("OUTPUT_PNG", "generated_trajectory.png")
>
> checkpoint_path = latest_checkpoint(config.checkpoints)
> print("Checkpoint:", checkpoint_path)
> print("Text:", text)
>
> model = HandwritingSynthesis(
>     vocab_size=config.vocab_size,
>     embed_dim=config.embed_dim,
>     lstm_size=config.lstm_size,
>     num_layers=config.num_lstm_layers,
>     K=config.K,
>     n_mixtures=config.n_mixtures,
>     kappa_initial_bias=config.kappa_initial_bias,
> ).to(config.device)
>
> checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
> model.load_state_dict(checkpoint["model_state_dict"])
> dxdy_mean = np.asarray(checkpoint.get("dxdy_mean", [[0.0, 0.0]]), dtype=np.float32)
> dxdy_std = np.asarray(checkpoint.get("dxdy_std", [[1.0, 1.0]]), dtype=np.float32)
>
> trajectory = generate(
>     model,
>     text,
>     config.char_to_idx,
>     max_len=max_len,
>     device=config.device,
>     bias=bias,
>     dxdy_mean=dxdy_mean,
>     dxdy_std=dxdy_std,
>     min_len=min_len,
> )
>
> with open(output_json, "w", encoding="utf-8") as f:
>     json.dump(trajectory, f, indent=2, ensure_ascii=False)
>
> xs, ys = [], []
> for x, y, e in trajectory:
>     xs.append(x)
>     ys.append(y)
>     if e == 1:
>         if len(xs) > 1:
>             plt.plot(xs, ys, "k-", linewidth=0.8)
>         xs, ys = [], []
> if len(xs) > 1:
>     plt.plot(xs, ys, "k-", linewidth=0.8)
> plt.gca().invert_yaxis()
> plt.axis("equal")
> plt.title(f'Generated: "{text}"')
> plt.savefig(output_png, dpi=180, bbox_inches="tight")
> print(f"Saved {len(trajectory)} points to {output_json}")
> print(f"Saved preview to {output_png}")
> PY

clean-npzs:
> rm -f dataset/npzs/*.npz dataset/all_trajectories.npz
