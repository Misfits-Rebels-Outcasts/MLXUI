# Fixtures/Gemma4 — provenance

Verbatim copies of two files from `mlx-community/gemma-4-e4b-it-4bit`
@ `475b9088d29754a3379866cf5aeb6b41acd313c2` (fetched 2026-10-21 by `RSI/notes/modelpack/fetch-hf.sh`).
They are tiny stand-ins for the 5.18 GB checkpoint, so MP1-3 and MP1-S unit tests can check what the real load will meet
without installing the model. **Never edit them to make a test pass**;
re-fetch from the pinned revision instead.

| File | Why a test reads it | sha256 |
|---|---|---|
| `chat_template.jinja` | renders through swift-jinja (plain chat, system turn, image placeholder) | `2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082` |
| `config.json` | the Local Server's memory admission, context window and Connect facts are read from it (MP1-S) | `780ccb3a514a5f1ced161d383f948fc22eca9b84b752ca19494f625bd9bad7a6` |

Used by `MLXUITests/Gemma4ResolutionTests.swift` (MP1-3) and `MLXUITests/CommentWhitespaceControlTests.swift` (MP1-3-FIX-1) and `MLXUITests/Gemma4ServeTests.swift` (MP1-S). Not part of the CatFlow fixture manifest.

## Generated: `prompt_goldens.json`

**Not** a verbatim copy: the template above rendered by Python's reference Jinja implementation, for four
conversations (one user turn; system + user; a six-message multi-turn; system + image + text; the Local Server conversation; tools declared; thinking on; and six assistant tool-call histories, MP1-S). Written by
`make_prompt_goldens.py` (`python3 -I Fixtures/Gemma4/make_prompt_goldens.py` from the repo root); the file
records its own provenance (Python 3.13.3, jinja2 3.1.6, `ImmutableSandboxedEnvironment(trim_blocks=True,
lstrip_blocks=True)`, the template's sha256, the model revision). **Never edit it to make a test pass**:
regenerate it, and if it changes, that is a finding. It exists because swift-jinja ignores the `-` on comment
tags; `ChatTemplateSanitizer` compensates and `CommentWhitespaceControlTests` proves the result is
byte-identical to these.
