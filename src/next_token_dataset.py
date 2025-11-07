# gpt2_model.py
# Простая обёртка для distilgpt2: загрузка, encode/decode, генерация.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class GPT2AutoCompleter:
    def __init__(self, model_name="distilgpt2", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        self.beam = dict(num_beams=4, early_stopping=True, length_penalty=1.0, no_repeat_ngram_size=3)
        self.nucleus = dict(do_sample=True, top_p=0.95, temperature=0.8, no_repeat_ngram_size=3)

    def encode(self, text, max_length=128):
        return self.tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_length)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens, preset="beam"):
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        kwargs = self.beam if preset == "beam" else self.nucleus
        out = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        return out[0].tolist()

    @torch.no_grad()
    def complete_last_quarter(self, full_ids):
        if len(full_ids) < 4:
            full_ids = (full_ids + full_ids)[:4]
        cut = max(1, min(len(full_ids) - 1, (len(full_ids) * 3 + 3) // 4))
        head, tail = full_ids[:cut], full_ids[cut:]
        out_ids = self.generate(head, max_new_tokens=len(tail), preset="beam")
        gen_tail = out_ids[-len(tail):]
        return head, tail, gen_tail
