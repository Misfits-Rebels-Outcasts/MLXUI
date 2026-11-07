#!/usr/bin/env python3
"""Regenerates `Fixtures/TranslateGemma/prompt_goldens.json`: the TranslateGemma chat template rendered by Python's
reference Jinja implementation.

Run from the repo root:  python3 -I Fixtures/TranslateGemma/make_prompt_goldens.py
Needs `jinja2` (the reference implementation). Reads only `Fixtures/TranslateGemma/chat_template.jinja`.
The environment matches transformers' `apply_chat_template`: an immutable sandbox with trim_blocks and lstrip_blocks
on, the same two options swift-transformers compiles with. `bos_token` is supplied by the tokenizer config when the app
renders; the template itself emits it, so the rendered string starts with exactly one `<bos>`.

Three requests, per MC2-3 Trap 3: EN→MS, ZH-Hans→EN, and one regional code written with an underscore (en_GB), which
the template normalises to `en-GB`.
"""
import hashlib
import json
from datetime import date
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

DIR = "Fixtures/TranslateGemma/"
MODEL = "mlx-community/translategemma-4b-it-4bit"
REVISION = "5788ec08c047f3f2e17808101b8d9566ac930d58"

template = open(DIR + "chat_template.jinja", encoding="utf-8").read()
sha = hashlib.sha256(template.encode("utf-8")).hexdigest()
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
compiled = env.from_string(template)

def case(name, source, target, text):
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "source_lang_code": source,
            "target_lang_code": target,
            "text": text,
        }],
    }]
    ctx = {"bos_token": "<bos>", "messages": messages, "add_generation_prompt": True}
    return {"name": name, "context": ctx, "expected": compiled.render(**ctx)}

cases = [
    case("a_en_to_ms", "en", "ms", "The weather is lovely today."),
    case("b_zh_hans_to_en", "zh-Hans", "en", "今天天气很好。"),
    # The regional-code case: the template does `.replace("_", "-")`, so en_GB renders as `(en-GB)`.
    case("c_en_gb_to_ms", "en_GB", "ms", "The weather is lovely today."),
]

goldens = {
    "provenance": {
        "generator": "Fixtures/TranslateGemma/make_prompt_goldens.py",
        "generated": date.today().isoformat(),
        "python": __import__("sys").version.split()[0],
        "jinja2": jinja2.__version__,
        "environment": "ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)",
        "template": "Fixtures/TranslateGemma/chat_template.jinja",
        "templateSHA256": sha,
        "model": f"{MODEL} @ {REVISION}",
    },
    "cases": cases,
}

open(DIR + "prompt_goldens.json", "w", encoding="utf-8").write(
    json.dumps(goldens, ensure_ascii=False, indent=2) + "\n")
print(f"wrote {DIR}prompt_goldens.json: {len(cases)} cases, sha256 {sha}")
