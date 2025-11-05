import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from datasets import load_dataset


class MaskedBertDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.samples = []

        for line in texts:
            token_ids = tokenizer.encode(line, max_length=max_length, add_special_tokens=False, truncation=True)

            if len(token_ids) < 2:
                continue

            x = token_ids[:-1]
            y = token_ids[1:]

            self.samples.append((x,y))
    
    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)





def main():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    val_df = pd.read_csv('data/val.csv')

    train_dataset = MaskedBertDataset(train_df['text'], tokenizer)
    val_dataset = MaskedBertDataset(val_df['text'], tokenizer)
    test_dataset = MaskedBertDataset(test_df['text'], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=3, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=3, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=3, shuffle=False)

    for i in range(min(3, len(test_dataset))):
        x, y = test_dataset[i]
        print(f"#{i} len(x)={len(x)}, len(y)={len(y)}")
        print("X:", tokenizer.decode(x.tolist(), skip_special_tokens=False))
        print("Y:", tokenizer.decode(y.tolist(), skip_special_tokens=False))
        print("-" * 40)

if __name__ == '__main__':
    main()