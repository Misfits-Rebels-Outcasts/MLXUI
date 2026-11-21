#!/usr/bin/env python3
"""MC5-3e tiny-tier reference: upstream's AceStepConditionEncoder (the REAL remote-code class from the pinned checkpoint) built with a TINY config,
seeded random weights, fp32 CPU. Saves config + `encoder.*` weights (same key names as the real turbo file) + inputs + outputs (+ lyric/timbre
sub-outputs for localisation). Two cases: all-valid masks, and masks with padding (exercises pack_sequences' stable sort).
Usage: oracle/ACE-Step-1.5/.venv/bin/python -I scripts/dump_cond_tiny.py <turbo_dir> <out.safetensors>"""
import json, os, sys, numpy as np, torch
from safetensors.numpy import save_file
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
turbo, out = sys.argv[1], sys.argv[2]
base = json.load(open(os.path.join(turbo, "config.json")))
tiny = dict(base, hidden_size=32, intermediate_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=8, text_hidden_dim=16,
            timbre_hidden_dim=8, num_lyric_encoder_hidden_layers=2, num_timbre_encoder_hidden_layers=2, num_hidden_layers=2,
            num_audio_decoder_hidden_layers=2, num_attention_pooler_hidden_layers=1, sliding_window=4, layer_types=["sliding_attention", "full_attention"],
            vocab_size=64, fsq_dim=32, max_position_embeddings=128, dtype="float32")
tiny.pop("auto_map", None)
Cfg = get_class_from_dynamic_module("configuration_acestep_v15.AceStepConfig", turbo)
Enc = get_class_from_dynamic_module("modeling_acestep_v15_turbo.AceStepConditionEncoder", turbo)
cfg = Cfg(**{k: v for k, v in tiny.items() if k not in ("architectures", "model_type", "transformers_version")}); cfg._attn_implementation = "eager"
torch.manual_seed(1234); enc = Enc(cfg).float().eval()
for p in enc.parameters(): torch.nn.init.normal_(p, std=0.2) if p.dim() > 1 else torch.nn.init.normal_(p, std=0.1)
T = {}
for k, v in enc.state_dict().items(): T["w." + k] = v.detach().numpy().astype(np.float32)
g = torch.Generator().manual_seed(7)
def case(name, Lt, Ll, tmask, lmask):
    th = torch.randn(1, Lt, 16, generator=g); lh = torch.randn(1, Ll, 16, generator=g); refer = torch.randn(1, 6, 8, generator=g)
    order = torch.zeros(1, dtype=torch.long)
    tm = torch.tensor([tmask], dtype=torch.bool); lm = torch.tensor([lmask], dtype=torch.bool)
    with torch.no_grad():
        eh, em = enc(text_hidden_states=th, text_attention_mask=tm, lyric_hidden_states=lh, lyric_attention_mask=lm,
                     refer_audio_acoustic_hidden_states_packed=refer, refer_audio_order_mask=order)
        lo = enc.lyric_encoder(inputs_embeds=lh, attention_mask=lm).last_hidden_state
        to = enc.timbre_encoder(refer, order)
        tp = enc.text_projector(th)
    T[f"{name}.in.text_hidden"] = th.numpy(); T[f"{name}.in.text_mask"] = tm.numpy().astype(np.uint8)
    T[f"{name}.in.lyric_embeds"] = lh.numpy(); T[f"{name}.in.lyric_mask"] = lm.numpy().astype(np.uint8)
    T[f"{name}.in.refer"] = refer.numpy(); T[f"{name}.in.refer_order"] = order.numpy().astype(np.int32)
    T[f"{name}.out.encoder_hidden_states"] = eh.numpy(); T[f"{name}.out.encoder_mask"] = em.numpy().astype(np.uint8)
    T[f"{name}.out.lyric_encoder"] = lo.numpy(); T[f"{name}.out.text_projector"] = tp.numpy()
    for i, t in enumerate(to if isinstance(to, tuple) else [to]): T[f"{name}.out.timbre.{i}"] = t.numpy().astype(np.float32) if t.dtype != torch.bool else t.numpy().astype(np.uint8)
case("valid", 5, 7, [1] * 5, [1] * 7)
case("padded", 6, 9, [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0])
meta = {"config": json.dumps(tiny), "note": "tiny-config upstream AceStepConditionEncoder, seed 1234 weights (normal 0.2/0.1), fp32 CPU eager attention, pinned ca1e85f / 19671f4",
        "pinned_upstream": "ca1e85fe9430179831e6bc6be790c332190a3866", "pinned_weights": "ACE-Step/Ace-Step1.5@19671f406d603126926c1b7e2adc169acbcade22", "torch": torch.__version__}
save_file(T, out, metadata=meta)
print(len(T), "tensors", os.path.getsize(out), "bytes")
for k in sorted(T):
    if not k.startswith("w."): print(k, T[k].shape)
print(sorted(k for k in T if k.startswith("w."))[:40])
