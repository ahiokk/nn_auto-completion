# eval_transformer_pipeline.py
import os, math, random
import pandas as pd
import evaluate
from tqdm import tqdm
from next_token_dataset import GPT2AutoCompleter

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

VAL_PATH = "data/val.csv"
TEXT_COL = "text"
LIMIT = int(os.getenv("EVAL_LIMIT", "20000"))
PRESET = os.getenv("EVAL_PRESET", "nucleus")
MAX_LEN = int(os.getenv("EVAL_MAX_LEN", "128"))
MAX_NEW_TOKENS_CAP = int(os.getenv("EVAL_CAP", "48"))

def split_three_quarter(ids):
    if len(ids) < 4:
        ids = (ids + ids)[:4]
    n = len(ids)
    cut = max(1, min(n - 1, math.ceil(0.75 * n)))
    return ids[:cut], ids[cut:]

def main():
    model = GPT2AutoCompleter("distilgpt2")
    tok = model.tokenizer
    try: model.model.gradient_checkpointing_disable()
    except Exception: pass
    if model.device == "cuda":
        model.model.half()

    df = pd.read_csv(VAL_PATH)
    texts = df[TEXT_COL].fillna("").tolist()
    if LIMIT and LIMIT > 0:
        texts = texts[:LIMIT]

    preds, refs = [], []
    ctxs = []                               

    for text in tqdm(texts, desc=f"Eval GPT2 ({PRESET}, cap={MAX_NEW_TOKENS_CAP})"):
        full_ids = tok.encode(str(text), add_special_tokens=False, truncation=True, max_length=MAX_LEN)
        head_ids, tail_ids = split_three_quarter(full_ids)
        if not tail_ids:
            continue
        need = min(len(tail_ids), MAX_NEW_TOKENS_CAP)
        out_ids = model.generate(head_ids, max_new_tokens=need, preset=PRESET)
        gen_tail = out_ids[-need:]

        preds.append(tok.decode(gen_tail,            skip_special_tokens=True, clean_up_tokenization_spaces=True))
        refs.append (tok.decode(tail_ids[:need],     skip_special_tokens=True, clean_up_tokenization_spaces=True))
        ctxs.append (tok.decode(head_ids,            skip_special_tokens=True, clean_up_tokenization_spaces=True))  # EXAMPLES

    rouge = evaluate.load("rouge")
    res = rouge.compute(
        predictions=[s.strip() for s in preds],
        references=[s.strip() for s in refs],
        use_stemmer=True
    )
    print("ROUGE:", {k: round(float(v), 4) for k, v in res.items()})


    good, bad = [], []
    for c, p, r in zip(ctxs, preds, refs):
        if p.strip().lower() == r.strip().lower():
            good.append((c, p, r))
        else:
            bad.append((c, p, r))

    random.seed(42)
    print("\nSome CORRECT predictions:")
    for c, p, r in random.sample(good, min(5, len(good))):
        print(f"CTX: {c}\nGEN: {p}\nREF: {r}\n" + "-"*60)

    print("\nSome INCORRECT predictions:")
    for c, p, r in random.sample(bad, min(5, len(bad))):
        print(f"CTX: {c}\nGEN: {p}\nREF: {r}\n" + "-"*60)

if __name__ == "__main__":
    main()
