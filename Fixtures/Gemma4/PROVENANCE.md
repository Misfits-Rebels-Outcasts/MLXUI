# Fixtures/Gemma4 — provenance

Verbatim copy of one file from `mlx-community/gemma-4-e4b-it-4bit`
@ `475b9088d29754a3379866cf5aeb6b41acd313c2` (fetched 2026-10-21 by `RSI/notes/modelpack/fetch-hf.sh`).
It is a tiny stand-in for the 5.18 GB checkpoint, so MP1-3's unit tests can check that the template
renders through swift-jinja without installing the model. **Never edit it to make a test pass**;
re-fetch from the pinned revision instead.

| File | Why a test reads it | sha256 |
|---|---|---|
| `chat_template.jinja` | renders through swift-jinja (plain chat, system turn, image placeholder) | `2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082` |

Used by `MLXUITests/Gemma4ResolutionTests.swift` (MP1-3) and `MLXUITests/CommentWhitespaceControlTests.swift` (MP1-3-FIX-1). Not part of the CatFlow fixture manifest.

## Generated: `prompt_goldens.json`

**Not** a verbatim copy: the template above rendered by Python's reference Jinja implementation, for four
conversations (one user turn; system + user; a six-message multi-turn; system + image + text). Written by
`make_prompt_goldens.py` (`python3 -I Fixtures/Gemma4/make_prompt_goldens.py` from the repo root); the file
records its own provenance (Python 3.13.3, jinja2 3.1.6, `ImmutableSandboxedEnvironment(trim_blocks=True,
lstrip_blocks=True)`, the template's sha256, the model revision). **Never edit it to make a test pass**:
regenerate it, and if it changes, that is a finding. It exists because swift-jinja ignores the `-` on comment
tags; `ChatTemplateSanitizer` compensates and `CommentWhitespaceControlTests` proves the result is
byte-identical to these.
