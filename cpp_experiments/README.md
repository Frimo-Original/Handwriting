# C++ Train-Step Benchmark

Этот эксперимент проверяет, сколько можно выиграть, если перенести один
train-step текущей Graves-style модели из Python в C++/LibTorch, не меняя саму
архитектуру.

Что измеряется:

- forward;
- loss;
- backward;
- optimizer step;
- общий train-step на одном подготовленном batch.

Что не измеряется:

- качество модели;
- полный training loop с чекпоинтами;
- генерация;
- загрузка всего датасета.

## Быстрый запуск

Из корня проекта:

```bash
.venv/bin/python cpp_experiments/run_benchmark.py \
  --device cpu \
  --batch-size 24 \
  --batch-mode median \
  --warmup 1 \
  --iters 5
```

На PC с CUDA:

```bash
python cpp_experiments/run_benchmark.py \
  --device cuda \
  --batch-size 24 \
  --batch-mode median \
  --warmup 1 \
  --iters 8
```

`run_benchmark.py` делает три шага:

1. Готовит `cpp_experiments/batches/batch.bin` из текущего датасета.
2. Собирает C++ benchmark через CMake и LibTorch из установленного PyTorch.
3. Запускает `bench_train_step`.

## Windows

Если запуск падает с `FileNotFoundError: cmake`, CMake не найден в `PATH`.
Самый простой вариант:

```powershell
python -m pip install cmake
```

После этого повторить команду benchmark. Также можно передать путь явно:

```powershell
python cpp_experiments/run_benchmark.py `
  --device cuda `
  --batch-size 24 `
  --batch-mode median `
  --warmup 1 `
  --iters 8 `
  --cmake "C:\Program Files\CMake\bin\cmake.exe"
```

Для сборки C++ на Windows также нужен MSVC compiler, обычно из Visual Studio
Build Tools с компонентом `Desktop development with C++`.

Если CMake падает на `CUDA::nvToolsExt`, обновите файлы эксперимента и
перезапустите команду. В `CMakeLists.txt` добавлен compatibility shim для новых
CUDA/NVTX, где старого target уже нет.

## Ручной запуск

Подготовить batch:

```bash
.venv/bin/python cpp_experiments/prepare_batch.py \
  --batch-size 24 \
  --max-seq-len 3000 \
  --mode median \
  --output cpp_experiments/batches/batch.bin
```

Собрать:

```bash
cmake -S cpp_experiments -B cpp_experiments/build \
  -DCMAKE_PREFIX_PATH="$(.venv/bin/python - <<'PY'
from torch.utils import cmake_prefix_path
print(cmake_prefix_path)
PY
)"

cmake --build cpp_experiments/build --config Release --parallel
```

Запустить:

```bash
cpp_experiments/build/bench_train_step \
  --batch cpp_experiments/batches/batch.bin \
  --device cpu \
  --vocab-size 90 \
  --lstm-size 400 \
  --K 10 \
  --n-mixtures 20 \
  --warmup 1 \
  --iters 5
```

На Windows исполняемый файл обычно лежит в:

```text
cpp_experiments/build/Release/bench_train_step.exe
```

## Как интерпретировать

Если C++ быстрее Python на 5-15%, значит Python overhead есть, но он не главный
узкий участок.

Если C++ быстрее на 25-30% и больше, можно думать о переносе отдельных горячих
частей.

Если разницы почти нет, проблема в последовательной recurrent-структуре и
тысячах маленьких операций, а не в языке Python.
