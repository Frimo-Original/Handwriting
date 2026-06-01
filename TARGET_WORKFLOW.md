# Target handwriting workflow

Цель проекта теперь не просто генерировать похожий почерк, а надежно получать
несколько заданных слов.

## 1. Задать целевые слова

Отредактируйте:

```text
dataset/target_texts.txt
```

Или передавайте слова напрямую:

```bash
.venv/bin/python scripts/handwriting.py candidates --texts "слово,мама,рама"
```

## 2. Обучать с валидацией

```bash
.venv/bin/python scripts/handwriting.py train --more-epochs 20
```

Обучение использует стабильный validation split из `src/config.py`.
При улучшении `val_loss` сохраняется:

```text
checkpoints_attention_eos_quotes/best.pth
```

Проверить текущий чекпойнт:

```bash
.venv/bin/python scripts/handwriting.py evaluate
```

## 3. Генерировать много кандидатов

```bash
.venv/bin/python scripts/handwriting.py candidates \
  --texts dataset/target_texts.txt \
  --output-dir runs/candidates/run_01
```

Скрипт сохраняет JSON, PNG и meta-диагностику по каждому варианту, а также:

```text
runs/candidates/run_01/candidates.csv
runs/candidates/run_01/best_candidates.json
```

Рейтинг пока эвристический: длина, завершенность attention, bbox, число отрывов
пера и уверенность MDN. Для почти безошибочного режима следующий сильный шаг -
добавить распознаватель/OCR или ручные оценки в CSV.

## 4. Один быстрый sample

```bash
.venv/bin/python scripts/handwriting.py generate --text "слово" --bias 1.25
```

Теперь рядом с JSON/PNG пишется `.meta.json` с диагностикой генерации.
