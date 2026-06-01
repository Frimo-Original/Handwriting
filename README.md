# Handwriting3.12

Проект генерирует рукописный текст как последовательность движений пера. На
вход модель получает строку, например `слово`, а на выходе создает траекторию:
список точек `[x, y, state]`, где `state = 0` означает продолжение штриха, а
`state = 1` означает отрыв пера.

Главная практическая цель сейчас - не “писать любые тексты”, а надежно писать
несколько заранее выбранных слов. Поэтому проект устроен вокруг короткого
цикла: собрать/проверить траектории, обучить модель, сгенерировать много
вариантов целевых слов и выбрать лучшие.

## Что внутри

Модель основана на статье Алекса Грейвса из `book.pdf`:

- LSTM хранит контекст уже нарисованной траектории.
- Soft-window attention двигается по символам заданного текста и подсказывает,
  какую букву модель сейчас должна писать.
- Mixture Density Network предсказывает распределение следующего смещения пера
  `dx, dy` и вероятность отрыва пера.

Модель не рисует картинку напрямую. Она генерирует онлайн-траекторию пера,
которую потом можно отрисовать в PNG или сохранить в JSON.

## С чем работаем

```text
dataset/jsons/           Исходные траектории: trajectory_<id>.json
dataset/texts/           Подписи к этим траекториям: trajectory_<id>.txt
dataset/all_trajectories.npz
                         Собранный датасет для обучения
checkpoints_attention/   Сохраненные веса модели
runs/                    Результаты генерации, оценки и отбора кандидатов
```

Одна обучающая пара состоит из:

```text
dataset/jsons/trajectory_120.json
dataset/texts/trajectory_120.txt
```

JSON хранит движения пера, TXT хранит текст, который написан этой траекторией.
Подробный формат датасета и скрипты разметки описаны в
`dataset/README.md`.

## Целевые слова

Целевые слова - это небольшой список слов/фраз, которые мы хотим научиться
писать особенно надежно. Они лежат здесь:

```text
dataset/target_texts.txt
```

Пример:

```text
слово
мама
рама
```

Этот файл сам по себе не обучает модель. Он используется для проверки и
генерации кандидатов: модель много раз пишет каждое слово с разными
случайными seed и bias, а проект сохраняет варианты для сравнения.

## Почему генерируем кандидатов

Генерация стохастическая: один запуск может получиться неудачным, даже если
модель в целом уже умеет писать слово. Поэтому основной режим качества - не
один `generate`, а `candidates`: создать много вариантов, посчитать диагностику
и выбрать лучшие.

Сейчас ранжирование эвристическое: учитываются длина траектории, прогресс
attention, размер bbox, число отрывов пера и уверенность MDN. Следующий шаг для
почти безошибочного режима - добавить OCR или ручную оценку в `candidates.csv`.

## Установка

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Проверить, что проект видит датасет и чекпойнты:

```bash
.venv/bin/python scripts/handwriting.py status
```

## Основной workflow

### 1. Подготовить целевые слова

Отредактировать:

```text
dataset/target_texts.txt
```

Начинать лучше с 2-5 слов. Если список слишком большой, будет сложнее понять,
что именно улучшилось или сломалось.

### 2. Проверить или пересобрать датасет

Если менялись `dataset/jsons/` или `dataset/texts/`, нужно пересобрать общий
`.npz`:

```bash
.venv/bin/python scripts/handwriting.py dataset
```

Это создает/обновляет:

```text
dataset/all_trajectories.npz
```

### 3. Оценить текущую модель

```bash
.venv/bin/python scripts/handwriting.py evaluate
```

Эта команда считает loss на валидационной части датасета. Она нужна не для
оценки красоты почерка глазами, а чтобы понимать, стало ли обучение лучше или
хуже относительно предыдущих запусков.

### 4. Дообучать маленькими шагами

```bash
.venv/bin/python scripts/handwriting.py train --more-epochs 20
```

Обучение продолжится с последнего чекпойнта. Если валидационная ошибка
улучшается, сохраняется:

```text
checkpoints_attention/best.pth
```

Обычные постоянные чекпойнты сохраняются каждые `save_every` эпох. При этом
включены временные промежуточные чекпойнты: после каждой эпохи сохраняется
`epoch_<n>.pth`, но если эта эпоха не кратна `save_every`, файл помечается как
temporary и удаляется после сохранения следующей эпохи. Так можно продолжить
обучение после сбоя, не захламляя папку чекпойнтов.

Не стоит сразу запускать сотни эпох: на маленьком датасете модель легко может
переучиться.

### 5. Сгенерировать кандидатов для целевых слов

```bash
.venv/bin/python scripts/handwriting.py candidates \
  --texts dataset/target_texts.txt \
  --output-dir runs/candidates/run_01
```

В результате появятся:

```text
runs/candidates/run_01/candidates.csv
runs/candidates/run_01/best_candidates.json
runs/candidates/run_01/<word>/*.png
runs/candidates/run_01/<word>/*.json
runs/candidates/run_01/<word>/*.meta.json
```

PNG нужны для просмотра глазами, JSON - это сами траектории, meta-файлы хранят
диагностику генерации.

### 6. Сравнить и повторить

Обычный цикл работы:

```bash
.venv/bin/python scripts/handwriting.py evaluate
.venv/bin/python scripts/handwriting.py candidates --output-dir runs/candidates/baseline
.venv/bin/python scripts/handwriting.py train --more-epochs 20
.venv/bin/python scripts/handwriting.py evaluate
.venv/bin/python scripts/handwriting.py candidates --output-dir runs/candidates/after_20
```

После этого сравниваются PNG и `candidates.csv`: стало ли слово читаемее,
меньше ли мусорных штрихов, стабильнее ли конец слова.

## Быстрая генерация одного слова

Для быстрой проверки можно создать один вариант:

```bash
.venv/bin/python scripts/handwriting.py generate --text "слово" --bias 1.25
```

Результаты сохраняются в:

```text
runs/single/
```

Этот режим удобен для быстрой проверки, но не является основным способом
добиваться надежности.

## Структура проекта

```text
src/                     Код модели, датасета, обучения, генерации и оценки
scripts/handwriting.py   Главный CLI для повседневной работы
scripts/server_*.py      Совместимость со старыми командами запуска
dataset/scripts/         Утилиты конвертации, просмотра и разметки датасета
runs/                    Генерации, запуски кандидатов и отчеты
reports/                 Материалы отчета/курсовой
book.pdf                 Основная статья, на которой основан проект
```

`src/` должен содержать только исходный код. Все результаты генерации должны
лежать в `runs/`.

## Полезные команды

```bash
# Состояние проекта
.venv/bin/python scripts/handwriting.py status

# Найти узкое место обучения на нескольких batch
.venv/bin/python scripts/handwriting.py profile-train --batches 8 --warmup 1

# Пересобрать датасет
.venv/bin/python scripts/handwriting.py dataset

# Обучать до конкретной эпохи
.venv/bin/python scripts/handwriting.py train --epochs 450

# Дообучить с меньшей скоростью обучения
.venv/bin/python scripts/handwriting.py train --more-epochs 20 --lr 0.00003

# Оставлять постоянный чекпойнт каждую эпоху
.venv/bin/python scripts/handwriting.py train --more-epochs 20 --save-every 1

# Сгенерировать больше кандидатов на каждый bias
.venv/bin/python scripts/handwriting.py candidates --variants 16

# Удалить старые сгенерированные файлы вне датасета/чекпойнтов
.venv/bin/python scripts/handwriting.py clean

# Также удалить runs/
.venv/bin/python scripts/handwriting.py clean --include-runs
```

## Что хранить в Git

Обычно стоит хранить:

- исходный код в `src/` и `scripts/`;
- `dataset/jsons/` и `dataset/texts/`;
- README-файлы и конфиги;
- небольшие фиксированные списки вроде `dataset/target_texts.txt`.

Обычно не стоит хранить:

- `runs/`;
- `dataset/npzs/`;
- `dataset/all_trajectories.npz`;
- новые чекпойнты `*.pth`, если они не нужны как часть релиза.
