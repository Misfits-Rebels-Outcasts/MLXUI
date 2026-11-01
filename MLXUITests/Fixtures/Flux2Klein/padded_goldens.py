#!/usr/bin/env python3
"""MP7-3c-FIX-1: tiny-config fp32 goldens for RIGHT-PADDED text-encoder input (g3c_textencoder.safetensors has none).

Same tiny Qwen3 as tiny_goldens.py (same seed, same constructor, same randomize), so the weights are the ones in g3c_textencoder.safetensors;
the script asserts that and writes NO weights, only inputs and outputs. Two cases at length 16: 5 real + 11 pad, 7 real + 9 pad.
HF Qwen3Model gets attention_mask (1 real / 0 pad) and output_hidden_states, exactly as the pipeline's _get_qwen3_prompt_embeds does; taps
(1, 2, 3) stand in for (9, 18, 27); outputs are stacked as the pipeline stacks them: (B, S, 3*H).

  .venv/bin/python padded_goldens.py --out outputs5      (run from klein-ref/; copy the one output file next to this script)
"""
import argparse, hashlib, json, os
import numpy as np, torch
from safetensors.torch import save_file, load_file

SEED = 20261031
TINY_QWEN = dict(vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=4, num_attention_heads=4,
                 num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6, rope_theta=1000000.0, max_position_embeddings=64,
                 tie_word_embeddings=False)
TAP_LAYERS = (1, 2, 3)
PAD_ID = 127
CASES = {"pad5_11": 5, "pad7_9": 7}
LENGTH = 16


def randomize(model):   # identical to tiny_goldens.py
    g = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.ndim == 1:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g) if ("norm" in n or "scale" in n) else 0.05 * torch.randn(p.shape, generator=g))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs5"); ap.add_argument("--g3c", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "g3c_textencoder.safetensors"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from transformers import Qwen3Config, Qwen3Model
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    qwen = Qwen3Model(Qwen3Config(**TINY_QWEN)).float().eval(); randomize(qwen)
    # The weights must be the ones already committed in g3c (this script writes none).
    g3c = load_file(a.g3c)
    for k, v in qwen.state_dict().items():
        assert torch.equal(v.float(), g3c["w." + k]), f"weight {k} differs from g3c_textencoder.safetensors"
    gen = torch.Generator().manual_seed(SEED + 2)
    out = {}
    for name, real in CASES.items():
        ids = torch.randint(0, PAD_ID, (1, LENGTH), generator=gen)       # ids 0..126, never the pad id
        ids[:, real:] = PAD_ID
        mask = torch.zeros(1, LENGTH, dtype=torch.long); mask[:, :real] = 1
        with torch.no_grad():
            o = qwen(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
        hs = o.hidden_states
        stacked = torch.stack([hs[i] for i in TAP_LAYERS], dim=1).permute(0, 2, 1, 3).reshape(1, LENGTH, -1)
        out[f"{name}.in.input_ids"] = ids.float(); out[f"{name}.in.mask"] = mask.float(); out[f"{name}.out.prompt_embeds"] = stacked
    path = os.path.join(a.out, "g3c_padded.safetensors")
    # one metadata key only (safetensors writes metadata from a hash map; 2+ keys vary the header bytes)
    save_file({k: v.detach().to(torch.float32).contiguous() for k, v in out.items()}, path,
              metadata={"meta": json.dumps({"config": TINY_QWEN, "taps": TAP_LAYERS, "pad_id": PAD_ID, "cases": CASES, "length": LENGTH}, sort_keys=True)})
    print(hashlib.sha256(open(path, "rb").read()).hexdigest(), os.path.getsize(path), path)


if __name__ == "__main__":
    main()
