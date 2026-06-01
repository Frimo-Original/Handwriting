from pathlib import Path

import torch


# -----------------------------------------------------------------------------
# Пути проекта
# -----------------------------------------------------------------------------
# Все пути строятся относительно корня проекта. Поэтому скрипты можно запускать
# из корневой папки без ручной передачи абсолютных путей.
project_root = Path(__file__).resolve().parents[1]
data_path = str(project_root / "dataset" / "all_trajectories.npz")
checkpoints = str(project_root / "checkpoints_attention_eos_quotes")
runs_dir = str(project_root / "runs")


# -----------------------------------------------------------------------------
# Алфавит текста
# -----------------------------------------------------------------------------
# Модель умеет писать только символы из CHAR_SET. Для кавычек используем "„" как
# нижнюю открывающую и "“" как верхнюю. Старые прямые кавычки нормализуются в
# "“", чтобы в алфавите не было двух разных верхних кавычек.
# Конец фрагмента - отдельный токен EOS_TOKEN, чтобы "\n" оставался обычным
# переносом строки.
CHAR_SET = "\n„“ абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ.,!?-;:()1234567890"
EOS_TOKEN = "<EOS>"
VOCAB_TOKENS = list(CHAR_SET) + [EOS_TOKEN]
char_to_idx = {token: i for i, token in enumerate(VOCAB_TOKENS)}
idx_to_char = {i: token for token, i in char_to_idx.items()}
vocab_size = len(VOCAB_TOKENS)
eos_token = EOS_TOKEN
eos_char = EOS_TOKEN  # Совместимое имя для старого кода генерации.
append_eos_to_dataset = True
append_eos_to_generation = True
strip_final_newline_before_eos = True
# Пока в датасете нет реальных примеров "„", автоматическую парную замену
# прямых кавычек на "„...“" лучше держать выключенной. Старые прямые кавычки
# при этом всё равно заменяются на "“".
normalize_generation_quotes = False


def normalize_upper_quotes(text):
    return text.replace('"', "“")


def normalize_training_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = normalize_upper_quotes(text)
    if strip_final_newline_before_eos and text.endswith("\n"):
        text = text[:-1]
    return text


def normalize_russian_quotes(text):
    result = []
    is_opening = True
    for ch in text:
        if ch == '"':
            result.append("„" if is_opening else "“")
            is_opening = not is_opening
        else:
            result.append(ch)
    return "".join(result)


def normalize_generation_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if normalize_generation_quotes:
        return normalize_russian_quotes(text)
    return normalize_upper_quotes(text)


def tokenize_text(text, append_eos=False, normalize=False, normalize_quotes=False):
    if normalize:
        text = normalize_training_text(text)
    if normalize_quotes:
        text = normalize_russian_quotes(text)
    tokens = list(text)
    if append_eos:
        tokens.append(EOS_TOKEN)
    return tokens


def encode_text(text, append_eos=False, normalize=False, normalize_quotes=False, unknown_token=" "):
    fallback = char_to_idx[unknown_token]
    return [
        char_to_idx.get(token, fallback)
        for token in tokenize_text(text, append_eos, normalize, normalize_quotes)
    ]


def decode_tokens(indices, skip_eos=False):
    tokens = []
    for idx in indices:
        token = idx_to_char[int(idx)]
        if skip_eos and token == EOS_TOKEN:
            continue
        tokens.append(token)
    return "".join(tokens)


# -----------------------------------------------------------------------------
# Архитектура модели
# -----------------------------------------------------------------------------
# Модель в стиле Graves:
# - рекуррентная часть предсказывает движение пера;
# - attention window отслеживает позицию в целевом тексте;
# - MDN-голова предсказывает смесь распределений для следующего движения пера.
embed_dim = vocab_size
lstm_size = 400
num_lstm_layers = 3
n_mixtures = 20  # Количество Gaussian mixtures для траектории пера.
K = 10  # Количество attention mixtures по символам входного текста.
kappa_initial_bias = -4.0  # Замедляет старт attention, полезно для коротких слов.


# -----------------------------------------------------------------------------
# Датасет и валидация
# -----------------------------------------------------------------------------
# max_seq_len сильно влияет на скорость: чем меньше значение, тем быстрее эпоха,
# но тем выше риск обрезать длинные рукописные примеры.
max_seq_len = 3000
validation_split = 0.12
validation_seed = 20260531


# -----------------------------------------------------------------------------
# Устройство
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Расписание обучения
# -----------------------------------------------------------------------------
# Эти значения соответствуют текущему быстрому профилю для GTX 1660:
# python scripts/handwriting.py train --more-epochs 10 --batch-size 24
# --grad-accum-steps 1 --lr 0.00003 --max-seq-len 3000 --bucket-by-length
# --eval-every 5
batch_size = 24
grad_accum_steps = 1
learning_rate = 0.00003

# num_epochs используется только если more_epochs = None. При прямом запуске
# python src/run_training.py обучение обычно продолжается на more_epochs эпох
# от последнего найденного чекпоинта.
num_epochs = 500
more_epochs = 10

grad_clip = 10.0
attention_loss_weight = 0.01
pen_up_pos_weight = "auto"
max_pen_up_pos_weight = 100.0


# -----------------------------------------------------------------------------
# Батчинг и загрузка данных
# -----------------------------------------------------------------------------
# bucket_by_length группирует примеры похожей длины. Это уменьшает padding и,
# по профайлеру, дает самый заметный прирост скорости на CUDA.
bucket_by_length = True
bucket_size_multiplier = 50

num_workers = 0
pin_memory = device.type == "cuda"
cache_prepared_dataset = True
reload_dataset_each_epoch = True


# -----------------------------------------------------------------------------
# Чекпоинты и продолжение обучения
# -----------------------------------------------------------------------------
# save_temporary_each_epoch сохраняет последнюю непостоянную эпоху.
# Каждая save_every эпоха остается постоянным чекпоинтом.
auto_resume = True
resume_checkpoint = None
save_every = 5
save_temporary_each_epoch = True
save_best = True
best_metric = "val_loss"
eval_every = 5


# -----------------------------------------------------------------------------
# Производительность и вывод прогресса
# -----------------------------------------------------------------------------
cudnn_benchmark = True
compile_model = False
compile_mode = "default"
compile_heartbeat_seconds = 15
empty_cache_each_epoch = True

tqdm_ascii = True
progress_mode = "live"  # "compact" печатает одну строку на эпоху.
progress_mininterval = 1.0
progress_ncols = 120


# -----------------------------------------------------------------------------
# Целевые слова
# -----------------------------------------------------------------------------
# Главная цель проекта - надежно генерировать несколько коротких слов.
# Скрипты с поддержкой TARGET_TEXTS могут переопределять этот список из env.
target_texts = [
    "слово",
    "мама",
    "рама",
]


# -----------------------------------------------------------------------------
# Генерация кандидатов
# -----------------------------------------------------------------------------
# Candidate search генерирует несколько вариантов с разными sampling bias,
# чтобы потом выбрать самые читаемые результаты.
generation_bias = 1.25
candidate_biases = [0.75, 1.0, 1.25, 1.5, 1.75]
candidate_variants_per_bias = 8
candidate_base_seed = 12345
candidate_max_len_per_char = 350
candidate_min_len_per_char = 25
candidate_output_dir = str(project_root / "runs" / "candidates" / "latest")
