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
OUTPUT_META ?= generated_trajectory.meta.json
STOP_STRATEGY ?= max
APPEND_EOS ?= false
TARGET_TEXTS ?=
CANDIDATE_DIR ?= candidate_runs/latest
CANDIDATE_BIASES ?=
CANDIDATE_VARIANTS ?=
VALIDATION_SPLIT ?=
EVAL_EVERY ?=

ifneq ($(strip $(CUDA_VISIBLE_DEVICES)),)
CUDA_PREFIX := CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)
else
CUDA_PREFIX :=
endif

.PHONY: help venv check dataset train train-log latest generate evaluate candidates clean-npzs

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
>   '  make evaluate                     Evaluate latest/best checkpoint on validation split' \
>   '  make candidates TARGET_TEXTS="..." Generate/rank many candidates for target words' \
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
> "$(PY)" scripts/handwriting.py dataset

latest:
> if [ -d "$(CHECKPOINTS)" ]; then find "$(CHECKPOINTS)" -maxdepth 1 -name 'epoch_*.pth' | sort -V | tail -n 1; else echo "Missing checkpoint dir: $(CHECKPOINTS)"; fi

train: dataset
> $(CUDA_PREFIX) "$(PY)" scripts/handwriting.py train --checkpoints "$(CHECKPOINTS)" --epochs "$(EPOCHS)" --more-epochs "$(MORE_EPOCHS)" --batch-size "$(BATCH_SIZE)" --grad-accum-steps "$(GRAD_ACCUM_STEPS)" --lr "$(LR)" --num-workers "$(NUM_WORKERS)" --save-every "$(SAVE_EVERY)" --max-seq-len "$(MAX_SEQ_LEN)" --validation-split "$(VALIDATION_SPLIT)" --eval-every "$(EVAL_EVERY)" --progress-mode "$(PROGRESS_MODE)" $(if $(filter false,$(AUTO_RESUME)),--no-auto-resume,) --resume "$(RESUME)"

train-log:
> $(MAKE) train 2>&1 | tee -a "$(LOG)"

generate: venv
> $(CUDA_PREFIX) "$(PY)" scripts/handwriting.py generate --checkpoints "$(CHECKPOINTS)" --text "$(TEXT)" --bias "$(BIAS)" --min-len "$(MIN_GEN_LEN)" --max-len "$(MAX_GEN_LEN)" --stop-strategy "$(STOP_STRATEGY)" --output-dir "$$(dirname "$(OUTPUT_JSON)")"

evaluate: venv
> $(CUDA_PREFIX) "$(PY)" scripts/handwriting.py evaluate --checkpoints "$(CHECKPOINTS)"

candidates: venv
> $(CUDA_PREFIX) "$(PY)" scripts/handwriting.py candidates --checkpoints "$(CHECKPOINTS)" --texts "$(TARGET_TEXTS)" --output-dir "$(CANDIDATE_DIR)" $(if $(strip $(CANDIDATE_BIASES)),--biases "$(CANDIDATE_BIASES)",) $(if $(strip $(CANDIDATE_VARIANTS)),--variants "$(CANDIDATE_VARIANTS)",)

clean-npzs:
> rm -f dataset/npzs/*.npz dataset/all_trajectories.npz
