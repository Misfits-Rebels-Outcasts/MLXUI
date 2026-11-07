#!/usr/bin/env python3
"""Regenerates `MLXUI/Modules/TranslateGemma/TranslateGemmaLanguages+Data.swift`.

Run from the repo root:  python3 -I Fixtures/TranslateGemma/make_languages.py
Reads only `Fixtures/TranslateGemma/chat_template.jinja` (the pinned copy) and writes the Swift table.

The table is the template's own `languages` dict: 581 codes → 161 display names (owner decision 6). The Translate
screen shows one entry per name, using the name's base code (the code with no region/script subtag when the template
has one; `Filipino` falls back to `fil-PH`, its only code). Chinese is the one exception: it is split into
`Chinese (Simplified)` → `zh-Hans` and `Chinese (Traditional)` → `zh-Hant`, because the output script differs.
"""
import hashlib
import json
import re
import sys
from datetime import date

DIR = "Fixtures/TranslateGemma/"
OUT = "MLXUI/Modules/TranslateGemma/TranslateGemmaLanguages+Data.swift"
MODEL_REVISION = "5788ec08c047f3f2e17808101b8d9566ac930d58"

FAVOURITES = ["English", "Malay", "Chinese (Simplified)", "Chinese (Traditional)",
              "Indonesian", "Tamil", "Japanese", "Korean"]

raw = open(DIR + "chat_template.jinja", encoding="utf-8").read()
sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()

# The dict literal sits between `{%- set languages = {` and the closing `-%}`. It is JSON-shaped apart from the
# trailing comma, so parse the pairs directly rather than importing a Jinja environment: this keeps the "every code
# and name comes from the template" claim checkable by eye.
start = raw.index("{%- set languages = {")
end = raw.index("-%}", start)
pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', raw[start:end])
codes = [(code, name) for code, name in pairs]
names_in_order = []
seen = set()
for code, name in codes:
    if name not in seen:
        seen.add(name)
        names_in_order.append(name)

assert len(codes) == 581, f"expected 581 codes, got {len(codes)}"
assert len(names_in_order) == 161, f"expected 161 names, got {len(names_in_order)}"

by_name = {}
for code, name in codes:
    by_name.setdefault(name, []).append(code)

def base_code(name):
    aliases = by_name[name]
    for alias in aliases:              # the template's own order, bare code first where present
        if "-" not in alias and "_" not in alias:
            return alias
    return min(aliases, key=len)       # Filipino: only fil-PH

entries = []                           # (name, code, aliases or None for the Chinese pair)
for name in names_in_order:
    if name == "Chinese":
        entries.append(("Chinese (Simplified)", "zh-Hans", None))
        entries.append(("Chinese (Traditional)", "zh-Hant", None))
    else:
        entries.append((name, base_code(name), by_name[name]))

chinese_codes = by_name["Chinese"]
for name, code, aliases in entries:
    assert (aliases is None and code in chinese_codes) or (aliases is not None and code in aliases)

missing_favourites = [f for f in FAVOURITES if f not in [e[0] for e in entries]]
assert not missing_favourites, missing_favourites

def swift_string(s):
    return json.dumps(s, ensure_ascii=False)

lines = []
lines.append("// GENERATED FILE — do not edit by hand.")
lines.append("//")
lines.append("// Regenerate with:  python3 -I Fixtures/TranslateGemma/make_languages.py")
lines.append("//")
lines.append(f"// Source: Fixtures/TranslateGemma/chat_template.jinja (verbatim from")
lines.append(f"// mlx-community/translategemma-4b-it-4bit @ {MODEL_REVISION}), sha256 {sha}.")
lines.append(f"// The template lists {len(codes)} language codes for {len(names_in_order)} display names; the Chinese name is")
lines.append("// split into Simplified/Traditional (owner decision 6), giving the table below its 162 entries.")
lines.append("")
lines.append("import Foundation")
lines.append("")
lines.append("nonisolated extension TranslateGemmaLanguages {")
lines.append("    /// Every code the pinned template's `languages` dict accepts, in the template's order, with its display")
lines.append("    /// name. The request sends only codes from this table; the template's lookup is silent on an unknown one.")
lines.append("    static let templateCodes: [(code: String, name: String)] = [")
for code, name in codes:
    lines.append(f"        ({swift_string(code)}, {swift_string(name)}),")
lines.append("    ]")
lines.append("")
lines.append("    /// One entry per display name, template order, with the Chinese name split in place. `code` is the code")
lines.append("    /// the UI lists show; `aliases` are every template code that names the same language.")
lines.append("    static let entries: [Language] = [")
for name, code, aliases in entries:
    if aliases is None:
        alias_list = [code]
    else:
        alias_list = aliases
    joined = ", ".join(swift_string(a) for a in alias_list)
    lines.append(f"        Language(name: {swift_string(name)}, code: {swift_string(code)}, "
                 f"aliases: [{joined}]),")
lines.append("    ]")
lines.append("")
lines.append("    /// The Chinese codes the split entries cover. The template lists one `Chinese` name for all of them;")
lines.append("    /// the two entries above split them by script so the output script is never a guess.")
lines.append("    static let chineseCodes: [String] = [")
for code in chinese_codes:
    lines.append(f"        {swift_string(code)},")
lines.append("    ]")
lines.append("}")
lines.append("")

open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print(f"wrote {OUT}: {len(codes)} codes, {len(entries)} entries, sha256 {sha}")
