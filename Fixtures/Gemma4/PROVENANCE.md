# Fixtures/Gemma4 — provenance

Verbatim copy of one file from `mlx-community/gemma-4-e4b-it-4bit`
@ `475b9088d29754a3379866cf5aeb6b41acd313c2` (fetched 2026-10-21 by `RSI/notes/modelpack/fetch-hf.sh`).
It is a tiny stand-in for the 5.18 GB checkpoint, so MP1-3's unit tests can check that the template
renders through swift-jinja without installing the model. **Never edit it to make a test pass**;
re-fetch from the pinned revision instead.

| File | Why a test reads it | sha256 |
|---|---|---|
| `chat_template.jinja` | renders through swift-jinja (plain chat, system turn, image placeholder) | `2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082` |

Used by `MLXUITests/Gemma4ResolutionTests.swift` (MP1-3). Not part of the CatFlow fixture manifest.
