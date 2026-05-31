from pathlib import Path

import torch

# Data
CHAR_SET = "\n\" абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ.,!?-;:()1234567890"
char_to_idx = {ch: i for i, ch in enumerate(CHAR_SET)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}
vocab_size = len(CHAR_SET)

# Model. These values follow the handwriting synthesis setup from Graves.
embed_dim = vocab_size
lstm_size = 400
num_lstm_layers = 3
n_mixtures = 20
K = 10

max_seq_len = 3000

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training
batch_size = 2
grad_accum_steps = 2
learning_rate = 0.0001
num_epochs = 500
grad_clip = 10.0
attention_loss_weight = 0.01
pen_up_pos_weight = "auto"
max_pen_up_pos_weight = 100.0
kappa_initial_bias = -4.0

# GTX 1660 friendly defaults: stable fp32 training, small batches, CUDA input pinning.
num_workers = 0
pin_memory = device.type == "cuda"
cudnn_benchmark = True
cache_prepared_dataset = True
compile_model = True
compile_mode = "default"
compile_heartbeat_seconds = 15
save_every = 5
empty_cache_each_epoch = True
tqdm_ascii = True
progress_mode = "live"  # "live" redraws one tqdm line for the current epoch; use "compact" for one line per epoch.
progress_mininterval = 1.0
progress_ncols = 120
reload_dataset_each_epoch = True
auto_resume = True
resume_checkpoint = None

# Paths
project_root = Path(__file__).resolve().parents[1]
data_path = str(project_root / "dataset" / "all_trajectories.npz")
checkpoints = str(project_root / "checkpoints_attention")
