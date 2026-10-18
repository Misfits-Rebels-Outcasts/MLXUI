# Fixtures/MiMo — provenance

Verbatim copies of three files from `mlx-community/MiMo-V2.6-Distill-Qwen-9B-OptiQ-4bit`
@ `60061fc7499d25ce238e0687154557e504ea690a` (fetched 2026-10-17 by `RSI/notes/mimo/fetch-hf.sh`).
They are tiny stand-ins for the 7.1 GB checkpoint, so MM3's unit tests can check what the real load
will meet without installing it. **Never edit these to make a test pass**; re-fetch from the pinned
revision instead.

| File | Why a test reads it | sha256 |
|---|---|---|
| `chat_template.jinja` | the MiMo template renders through swift-jinja with tools and `enable_thinking: false` | `59a64ebb4df6d1489d09a91267cf3ceb106162d4a893c4f84833cfb8c897ff63` |
| `config.json` | the OptiQ per-layer `quantization` map decodes through `BaseConfiguration` | `6eca30db5b0d68c31eb699ba6b9da382d5e5c39915417df88aeb63acb3516294` |
| `model.safetensors.index.json` | its `.scales` tensors are exactly the override keys; shard layout for the file selector | `2225ecf6ae3a86b10e7156b729a355311d79fefd8fc49c3faff6002b89cdc435` |

Used by `MLXUITests/MiMoRunPathTests.swift` (MM3). Not part of the CatFlow fixture manifest.
