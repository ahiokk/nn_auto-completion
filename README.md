# Автодополнение текста: LSTM vs DistilGPT-2

Задача: по первым **3/4** текста сгенерировать оставшуюся **1/4** и сравнить модели по **ROUGE** и по примерам.

## Структура проекта

```
data/         train.csv  val.csv  test.csv
models/       lstm_best.pt  lstm_last.pt
src/
  data_utils.py                # Обработка датасета
  lstm_model.py                # код lstm модели
  lstm_train.py                # код обучения модели
  eval_lstm.py                 # замер метрик lstm модели
  eval_transformer_pipeline.py # код с запуском и замером качества трансформера
  next_token_dataset.py        # GPT-2 обёртка (генерация)
  split_dataset.py             # разделение исходного .txt файла на датасеты train/val/test
solution.ipynb                 # воспроизводимый ноутбук
requirements.txt
```
Важно! split_dataset нужно


## Установка

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python -m nltk.downloader punkt
```

### Рекомендуемые переменные окружения

```bash
export TOKENIZERS_PARALLELISM=false
# для ускорения оценки GPT-2:
export EVAL_LIMIT=20000      # сколько примеров валидки
export EVAL_PRESET=nucleus   # nucleus | beam
export EVAL_MAX_LEN=128      # длина входа
export EVAL_CAP=48           # макс. новых токенов
```

## Обучение LSTM

```bash
python src/lstm_train.py
```

* Чекпойнты сохраняются в `models/`:

  * `lstm_last.pt` — последний,
  * `lstm_best.pt` — лучший по `val_loss`.

### Только валидация/ROUGE без переобучения

```bash
export N_EPOCHS=0
python src/lstm_train.py
```

## Оценка DistilGPT-2 (без обучения)

```bash
python src/eval_transformer_pipeline.py
```

Скрипт берёт `val.csv`, режет 3/4 -> 1/4, генерирует хвост и считает **ROUGE**.
Параметры скорости/качества регулируются переменными `EVAL_*` (см. выше).

## Быстрый ROUGE из ноутбука (на подвыборке)

```python
import evaluate
rouge = evaluate.load("rouge")
res = rouge.compute(predictions=[p.strip() for p in preds[:20000]],
                    references=[r.strip() for r in refs[:20000]],
                    use_stemmer=True)
{k: round(float(v), 4) for k, v in res.items()}
```

## Как загрузить лучший чекпойнт LSTM

```python
import torch
from src.lstm_model import LSTMLastWord
from transformers import BertTokenizerFast

tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = LSTMLastWord(tok.vocab_size, emb_dim=128, hidden_dim=256)
ckpt = torch.load("models/lstm_best.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

## Выводы (кратко)

* **LSTM (текущая версия)** предсказывает **один токен** -> разумная метрика: **accuracy**, а ROUGE-2 технически ≈0.
* **DistilGPT-2** генерирует фразы -> ожидаемо **выше ROUGE-1/L** и **ненулевой ROUGE-2**; качество текстов лучше по примерам.
* Рекомендация: для человекочитаемого дописывания 1/4 текста — **DistilGPT-2**; для сверхбыстрых «подсказок следующего токена» — **LSTM**.

## Репликация в ноутбуке

Открой `solution.ipynb` и последовательно выполни ячейки:

1. проверка окружения и путей;
2. запуск LSTM (обучение/валидация, сохранение весов);
3. подсчёт ROUGE;
4. запуск DistilGPT-2 с параметрами;
5. итоговые выводы и примеры.

## Небольшие советы

* На Windows при `num_workers>0` добавляй защиту:

  ```python
  if __name__ == "__main__":
      import multiprocessing as mp
      mp.freeze_support()
      main()
  ```
* Если видишь ворнинги HF tokenizers — добавь `TOKENIZERS_PARALLELISM=false`.
* Для T4 16 ГБ попробуй `batch_size` 128–256 у LSTM и `EVAL_CAP` 48–64 у GPT-2.

