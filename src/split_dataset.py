import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split



def from_txt_to_csv(path):

    os.makedirs("data", exist_ok=True)

    out_path = "data/raw_data.csv"

    # читаем все строки
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f]

    # делаем DataFrame с одним столбцом text
    df = pd.DataFrame({"text": lines})

    # сохраняем
    df.to_csv(out_path, index=False)
    print(f"OK: {len(df)} строк сохранено в {out_path}")


def clean_text(text):

    text = text.lower()  # к нижнему регистру
    text = re.sub(r"[^a-z0-9 ]+", " ", text)  # оставить только буквы и цифры
    text = re.sub(r"\s+", " ", text).strip()  # убрать дублирующиеся пробелы

    return text

def cleaning_dataset(dataset):
    try:
        texts = dataset['text']
        texts = [clean_text(text) for text in texts]
    except:
        print('ошибка в очистке данных')
    
    out_path = "data/dataset_processed.csv"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pd.DataFrame({"text": texts}).to_csv(out_path, index=False)
    print('Датасет очищен')
    return texts


def split_80_10_10(dataset):
    #разделяем датсет на 80% train 20% temp
    train_df, temp_df = train_test_split(
        dataset,
        test_size=0.20,
        shuffle=True
    )

    #разделяем 20% temp на 10% val, 10% test
    val_df, test_df, = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        shuffle=True
    )

    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print(f'Длина train - {len(train_df)} \nДлина val - {len(val_df)} \nДлина test - {len(test_df)}')

def main():
    source_path = "data/tweets.txt"
    from_txt_to_csv(source_path)
    dataset = pd.read_csv('data/raw_data.csv')
    cleaning_dataset(dataset)
    split_80_10_10(dataset)
    print('Датасет успешно разделен и очищен')

if __name__ == '__main__':
    main()