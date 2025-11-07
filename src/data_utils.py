import torch
import re
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from torch.nn.utils.rnn import pad_sequence
from functools import partial

class MaskedBertDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.samples = []

        for line in texts:

            # приведение к нижнему регистру
            line = line.lower()
            # удаление всего, кроме латинских букв, цифр и пробелов
            line = re.sub(r'[^a-z0-9\s]', '', line)
            # удаление дублирующихся пробелов, удаление пробелов по краям
            line = re.sub(r'\s+', ' ', line).strip()
            
            token_ids = tokenizer.encode(line, max_length=max_length, add_special_tokens=False, truncation=True)

            if len(token_ids) < 2:
                continue

            x = token_ids[:-1]
            y = token_ids[1:]
            y_last = y[-1]

            self.samples.append((x,y))
    
    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)



def collate_last_word(batch, pad_id):
    xs, ys = zip(*batch)  # тензоры разной длины
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    x_pad = pad_sequence(xs, batch_first=True, padding_value=pad_id)  # (B, T_max)
    y_last = torch.tensor([int(y[-1]) for y in ys], dtype=torch.long) # цель = последний токен из y
    return x_pad, lengths, y_last




def data_utils():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_df = pd.read_csv('data/train.csv')
    test_df  = pd.read_csv('data/test.csv')
    val_df   = pd.read_csv('data/val.csv')

    pad_id = tokenizer.pad_token_id or 0

    train_dataset = MaskedBertDataset(train_df['text'], tokenizer)
    val_dataset   = MaskedBertDataset(val_df['text'],   tokenizer)
    test_dataset  = MaskedBertDataset(test_df['text'],  tokenizer)

    collate_fn = partial(collate_last_word, pad_id=pad_id) #нужно для работы num_workers

    train_loader = DataLoader(
        train_dataset, batch_size=64, num_workers=3, shuffle=True,
        collate_fn=collate_fn  
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, num_workers=3, shuffle=False,
        collate_fn=collate_fn  
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, num_workers=3, shuffle=False,
        collate_fn=collate_fn  
    )

    return train_loader, val_loader, test_loader, tokenizer
