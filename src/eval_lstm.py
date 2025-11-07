import evaluate
import numpy as np
import random

def compute_and_print_rouge(predictions, references, use_stemmer=True):
    assert len(predictions) == len(references), "pred/references sizes differ"
    rouge = evaluate.load("rouge")
    res = rouge.compute(                             
        predictions=[str(s).strip() for s in predictions],
        references=[str(s).strip() for s in references],
        use_stemmer=use_stemmer
    )
    print("ROUGE:")
    for k, v in res.items():
        print(f"{k}: {v:.4f}")
    return res
