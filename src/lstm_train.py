import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from data_utils import MaskedBertDataset, data_utils
from lstm_model import LSTMLastWord
from eval_lstm import compute_and_print_rouge

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  

MODELS_DIR = os.getenv("MODELS_DIR", "models") # Lля сохранения весов


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def evaluate(model, loader, criterion, device):  # GPU
    """Оценка без побочных print; возвращаем средний лосс и accuracy."""
    model.eval()
    correct, total = 0, 0
    sum_loss = 0.0
    with torch.no_grad():
        for x_pad, lengths, y_last in loader:
            # GPU
            x_pad   = x_pad.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            y_last  = y_last.to(device, non_blocking=True)

            x_output = model(x_pad, lengths)
            loss = criterion(x_output, y_last)
            preds = torch.argmax(x_output, dim=1)
            correct += (preds == y_last).sum().item()
            total   += y_last.size(0)
            sum_loss += loss.item()
    avg_loss = sum_loss / max(1, len(loader))
    accuracy = correct / max(1, total)
    return avg_loss, accuracy



def main():
    # GPU Настройка устройства и ускорителей
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU
    torch.backends.cudnn.benchmark = True                                   # GPU (ускорение для фикс. размеров)

    train_loader, val_loader, test_loader, tokenizer = data_utils()

    vocab_size = tokenizer.vocab_size
    emb_dim = 128
    hidden_dim = 256

    model = LSTMLastWord(vocab_size, emb_dim, hidden_dim).to(device)        # GPU 
    param_count = count_parameters(model)
    print('LSTM parameters count - ', param_count)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()


    n_epochs = 3
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for x_pad, lengths, y_last in tqdm(train_loader):
            # GPU:
            x_pad   = x_pad.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            y_last  = y_last.to(device, non_blocking=True)

            optimizer.zero_grad()

            x_output = model(x_pad, lengths)
            loss = criterion(x_output, y_last)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)  # GPU
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%}")


        last_ckpt_path = os.path.join(MODELS_DIR, "lstm_last.pt")  # SAVE
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
            },
            last_ckpt_path,
        )  # SAVE
        best_val_loss = float('inf')
        # Сохраняем лучший по val_loss
        if val_loss < best_val_loss: 
            best_val_loss = val_loss  
            best_ckpt_path = os.path.join(MODELS_DIR, "lstm_best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                },
                best_ckpt_path,
            )  
            print(f" Сохранен чекпоинт в: {best_ckpt_path} (val_loss={val_loss:.4f})")

    model.eval()
    bad_cases, good_cases = [], []
    gen, ref = [], []

    with torch.no_grad():
        for x_pad, lengths, y_last in val_loader:
            # GPU: перенос батча на устройство
            x_pad   = x_pad.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            y_last  = y_last.to(device, non_blocking=True)

            logits = model(x_pad, lengths)
            preds = torch.argmax(logits, dim=1)
            B = x_pad.size(0)
            for i in range(B):
                ctx_ids = x_pad[i, :lengths[i]].tolist()
                input_tokens = tokenizer.convert_ids_to_tokens(ctx_ids)
                true_tok = tokenizer.convert_ids_to_tokens([int(y_last[i].item())])[0]
                pred_tok = tokenizer.convert_ids_to_tokens([int(preds[i].item())])[0]

                gen.append(pred_tok)
                ref.append(true_tok)
                (bad_cases if preds[i] != y_last[i] else good_cases).append((input_tokens, true_tok, pred_tok))

    random.seed(42)
    bad_cases_sampled  = random.sample(bad_cases,  min(5, len(bad_cases)))
    good_cases_sampled = random.sample(good_cases, min(5, len(good_cases)))

    print("\nSome incorrect predictions:")
    for context, true_tok, pred_tok in bad_cases_sampled:
        print(f"Input: {' '.join(context)} | True: {true_tok} | Predicted: {pred_tok}")

    print("\nSome correct predictions:")
    for context, true_tok, pred_tok in good_cases_sampled:
        print(f"Input: {' '.join(context)} | True: {true_tok} | Predicted: {pred_tok}")

    compute_and_print_rouge(gen[:20000], ref[:20000])


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
