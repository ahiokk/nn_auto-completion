import torch
import torch.nn as nn
from transformers import BertTokenizerFast
from data_utils import MaskedBertDataset
import pandas as pd
from torch.utils.data import DataLoader
from lstm_model import LSTMLastWord


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def collate_last_word(batch, pad_id):
    xs, ys = zip(*batch)                       # xs: tuple[T_i], ys: tuple[scalar]
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    T = int(lengths.max().item())
    B = len(xs)
    x_pad = torch.full((B, T), pad_id, dtype=torch.long)
    for i, x in enumerate(xs):
        x_pad[i, :len(x)] = x
    y_last = torch.stack(ys)
    return x_pad, lengths, y_last


def main():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    pad_id = tokenizer.pad_token_id

    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')

    train_ds = MaskedBertDataset(train_df['text'], tokenizer)
    val_ds = MaskedBertDataset(val_df['text'], tokenizer)


    collate_fn = lambda b: collate_last_word(b, pad_id)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=2, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = LSTMLastWord(vocab_size=tokenizer.vocab_size, emb_dim=128, hidden_dim=256)
    crit = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=2e-3)


    for epoch in range(3):
        model.train()
        total_loss = 0.0
        for x_pad, lengths, y_last in train_loader:
            x_pad = x_pad.to(device)
            lengths = lengths.to(device)
            y_last = y_last.to(device)

            logits = model(x_pad, lengths)
            loss = crit(logits, y_last)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()


        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_pad, lengths, y_last in val_loader:
                x_pad   = x_pad.to(device)
                lengths = lengths.to(device)
                y_last  = y_last.to(device)
                preds = model(x_pad, lengths).argmax(dim=-1)
                correct += (preds == y_last).sum().item()
                total   += y_last.size(0)

        print(f"Epoch {epoch+1}: train_loss={total_loss/len(train_loader):.4f}  val_acc={correct/total:.4f}")

if __name__ == "__main__":
    main()
