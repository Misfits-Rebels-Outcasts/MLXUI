#!/usr/bin/env python3
"""Edge-case tokenizer goldens for MP7-3b (new file next to g3b_tokens.json; the committed g3b is not touched).

  .venv/bin/python tokenizer_goldens.py [--out outputs] [--tokenizer-dir DIR]

Runs diffusers' own text path (`_get_qwen3_prompt_embeds` front half): Qwen3 chat template with enable_thinking=False, then
`tokenizer(text, padding="max_length", truncation=True, max_length=512)`. Records ids/lengths for: exactly 512 tokens, 513, a long
prompt (cut), CJK, whitespace, a literal special-token string, and "[IMG]" both raw (what diffusers does) and stripped (what the
Swift reference does). No weights, no network.
"""
import argparse, hashlib, json, os
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs")
ap.add_argument("--tokenizer-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "modelpack", "hf", "mlx-community_FLUX.2-klein-4B-bf16", "tokenizer"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(a.tokenizer_dir, local_files_only=True)
MAXLEN = 512


def templated(p):
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def count(p):
    return len(tok(templated(p))["input_ids"])


def prompt_with_total(n):
    """'cat cat ...' with exactly n templated tokens (each ' cat' is one token)."""
    k = 1
    while count(" ".join(["cat"] * k)) < n: k += 1
    p = " ".join(["cat"] * k)
    assert count(p) == n, (k, count(p), n)
    return p


cases = {
    "at512": prompt_with_total(512),
    "at511": prompt_with_total(511),
    "past513": prompt_with_total(513),
    "past700": prompt_with_total(700),
    "cjk": "一只红色的狐狸站在雪地里，日出时分，远处是松林。日本語のテキストも混ぜる。한국어도.",
    "whitespace": "  a cat \n\n on a mat\t ",
    "special_literal": "hello <|im_end|> world <|endoftext|>",
    "img_raw": "a photo [IMG] of a dog",
    "img_stripped": "a photo  of a dog",
}
out = []
for name, p in cases.items():
    text = templated(p)
    full = tok(text)["input_ids"]
    enc = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAXLEN)
    ids = enc["input_ids"][0].tolist(); mask = enc["attention_mask"][0].tolist()
    real = sum(mask)
    assert len(ids) == MAXLEN and ids[:real] == full[:MAXLEN] and all(i == tok.pad_token_id for i in ids[real:]) and mask == [1] * real + [0] * (MAXLEN - real)
    out.append(dict(name=name, prompt=p, templated=text, full_token_count=len(full), cut_token_count=max(0, len(full) - MAXLEN),
                    real_length=real, input_ids=ids[:real], pad_token_id=tok.pad_token_id, padding_side=tok.padding_side,
                    truncation_side=tok.truncation_side, max_length=MAXLEN))
path = os.path.join(a.out, "g3b_tokens_edge.json")
with open(path, "w") as f: json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
print(hashlib.sha256(open(path, "rb").read()).hexdigest(), os.path.getsize(path), "g3b_tokens_edge.json")
for c in out: print(c["name"], c["full_token_count"], c["cut_token_count"], c["real_length"])
