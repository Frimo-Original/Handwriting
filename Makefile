SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.RECIPEPREFIX := >
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
> "$(PY)" scripts/server_check.py

dataset: venv
> mkdir -p dataset/npzs
> "$(PY)" dataset/converter.py

latest:
> if [ -d "$(CHECKPOINTS)" ]; then find "$(CHECKPOINTS)" -maxdepth 1 -name 'epoch_*.pth' | sort -V | tail -n 1; else echo "Missing checkpoint dir: $(CHECKPOINTS)"; fi

train: dataset
> $(CUDA_PREFIX) CHECKPOINTS="$(CHECKPOINTS)" EPOCHS="$(EPOCHS)" MORE_EPOCHS="$(MORE_EPOCHS)" BATCH_SIZE="$(BATCH_SIZE)" GRAD_ACCUM_STEPS="$(GRAD_ACCUM_STEPS)" LR="$(LR)" NUM_WORKERS="$(NUM_WORKERS)" SAVE_EVERY="$(SAVE_EVERY)" MAX_SEQ_LEN="$(MAX_SEQ_LEN)" PROGRESS_MODE="$(PROGRESS_MODE)" AUTO_RESUME="$(AUTO_RESUME)" RESUME="$(RESUME)" "$(PY)" scripts/server_train.py

train-log:
> $(MAKE) train 2>&1 | tee -a "$(LOG)"

generate: venv
> $(CUDA_PREFIX) CHECKPOINTS="$(CHECKPOINTS)" TEXT="$(TEXT)" BIAS="$(BIAS)" MIN_GEN_LEN="$(MIN_GEN_LEN)" MAX_GEN_LEN="$(MAX_GEN_LEN)" OUTPUT_JSON="$(OUTPUT_JSON)" OUTPUT_PNG="$(OUTPUT_PNG)" "$(PY)" scripts/server_generate.py

clean-npzs:
> rm -f dataset/npzs/*.npz dataset/all_trajectories.npz
