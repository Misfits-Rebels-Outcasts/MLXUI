#!/usr/bin/env python3
"""Regenerates prompt_goldens.json: the Gemma 4 chat template rendered by Python's jinja2.

Run from the repo root:  python3 -I Fixtures/Gemma4/make_prompt_goldens.py
Needs `jinja2` (the reference implementation). Reads only Fixtures/Gemma4/chat_template.jinja.
The environment matches transformers' `apply_chat_template`: an immutable sandbox with
trim_blocks and lstrip_blocks on, the same two options swift-transformers compiles with.
"""
import hashlib, json, sys
from datetime import date
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

DIR = "Fixtures/Gemma4/"
template = open(DIR + "chat_template.jinja", encoding="utf-8").read()
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
compiled = env.from_string(template)

def case(name, messages, **extra):
    ctx = {"bos_token": "<bos>", "messages": messages, "add_generation_prompt": True, **extra}
    return {"name": name, "context": ctx, "expected": compiled.render(**ctx)}

img = [{"type": "image"}, {"type": "text", "text": "What is this?"}]
cases = [
    case("a_one_user_turn", [{"role": "user", "content": "Hello"}]),
    case("b_system_and_user", [{"role": "system", "content": "Be brief."},
                               {"role": "user", "content": "Name three rivers in Europe."}]),
    case("c_multi_turn", [{"role": "system", "content": "You are a careful assistant."},
                          {"role": "user", "content": "What is 2+2?"},
                          {"role": "assistant", "content": "4."},
                          {"role": "user", "content": "And times 3?"},
                          {"role": "assistant", "content": "12."},
                          {"role": "user", "content": "Thanks. One more: 5 squared?"}]),
    case("d_system_image_text", [{"role": "system", "content": "Be brief."},
                                 {"role": "user", "content": img}]),
]
out = {
    "provenance": {
        "generator": "Fixtures/Gemma4/make_prompt_goldens.py",
        "generated": date.today().isoformat(),
        "python": sys.version.split()[0],
        "jinja2": jinja2.__version__,
        "environment": "ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)",
        "template": "Fixtures/Gemma4/chat_template.jinja",
        "templateSHA256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "model": "mlx-community/gemma-4-e4b-it-4bit @ 475b9088d29754a3379866cf5aeb6b41acd313c2",
    },
    "cases": cases,
}
json.dump(out, open(DIR + "prompt_goldens.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
open(DIR + "prompt_goldens.json", "a").write("\n")
