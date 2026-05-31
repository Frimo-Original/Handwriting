# Dataset

Эта папка хранит исходные рукописные траектории, текстовые подписи к ним и
сгенерированный `.npz`-датасет для обучения модели.

## Структура

```text
dataset/
  jsons/                  Исходные траектории: trajectory_<id>.json
  texts/                  Подписи: trajectory_<id>.txt
  scripts/                Утилиты для сбора, просмотра и разметки датасета
  npzs/                   Промежуточные .npz-файлы, генерируются автоматически
  all_trajectories.npz    Итоговый датасет для src/dataset.py
  target_texts.txt        Слова для режима candidate generation
  trash/                  Старые/ручные тестовые файлы
```

`jsons/` и `texts/` - это источник данных. `npzs/` и
`all_trajectories.npz` - build artifacts: их можно пересобрать.

## Формат trajectory JSON

Каждый `dataset/jsons/trajectory_<id>.json` содержит список точек:

```json
[
  [x, y, state],
  [x, y, state]
]
```

Где:

- `x`, `y` - координаты пера/курсора.
- `state = 0` - перо продолжает текущий штрих.
- `state = 1` - конец штриха / pen-up.

Файл `dataset/texts/trajectory_<id>.txt` должен содержать текст, который
написан в соответствующей траектории.

## Сборка датасета

Основной способ из корня проекта:

```bash
.venv/bin/python scripts/handwriting.py dataset
```

Прямой запуск конвертера:

```bash
.venv/bin/python dataset/scripts/converter.py
```

Конвертер:

1. Читает пары `jsons/trajectory_<id>.json` и `texts/trajectory_<id>.txt`.
2. Кодирует текст в индексы символов из `src/config.py`.
3. Пишет промежуточные файлы в `dataset/npzs/`.
4. Собирает общий `dataset/all_trajectories.npz`.

## Скрипты

```text
dataset/scripts/converter.py
```

Собирает `all_trajectories.npz` из `jsons/` и `texts/`. Это основной скрипт,
который нужен перед обучением после изменения исходных данных.

```text
dataset/scripts/label_trajectories.py
```

Tkinter/Matplotlib-интерфейс для просмотра траекторий и создания/правки
подписей в `dataset/texts/`. По умолчанию открывается в режиме просмотра.

```bash
.venv/bin/python dataset/scripts/label_trajectories.py --review
.venv/bin/python dataset/scripts/label_trajectories.py --edit
.venv/bin/python dataset/scripts/label_trajectories.py --start 120 --edit
```

```text
dataset/scripts/handrwritting.py
```

Старый Tkinter-редактор для записи траектории мышью/пером. Сейчас сохраняет
только JSON в `dataset/jsons/`, без текстовой подписи. После него нужно вручную
создать соответствующий `dataset/texts/trajectory_<id>.txt` или открыть
`label_trajectories.py --edit` и подписать новый образец там.

## Добавление новых образцов

1. Создать `dataset/jsons/trajectory_<new_id>.json`.
2. Создать `dataset/texts/trajectory_<new_id>.txt` с точной подписью вручную
   или через:

```bash
.venv/bin/python dataset/scripts/label_trajectories.py --start <new_id> --edit
```

3. Пересобрать датасет:

```bash
.venv/bin/python scripts/handwriting.py dataset
```

4. Проверить состояние проекта:

```bash
.venv/bin/python scripts/handwriting.py status
```

## Целевые слова

`dataset/target_texts.txt` не участвует в обучении напрямую. Это список слов или
коротких фраз для генерации и ранжирования кандидатов:

```bash
.venv/bin/python scripts/handwriting.py candidates \
  --texts dataset/target_texts.txt \
  --output-dir runs/candidates/run_01
```

## Что хранить в Git

Обычно стоит хранить:

- `dataset/jsons/`
- `dataset/texts/`
- `dataset/scripts/`
- `dataset/README.md`
- `dataset/target_texts.txt`

Обычно не стоит хранить:

- `dataset/npzs/`
- `dataset/all_trajectories.npz`
- временные файлы из `dataset/trash/`
