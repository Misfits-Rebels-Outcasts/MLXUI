# Qwen3-VL Embedding 2B parity fixtures (MC4-3)

Generated **once** on 2026-11-07 by `make_qwen3vl_embed_golden.py` from the **official bf16 reference**
`Qwen/Qwen3-VL-Embedding-2B` at revision `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` (its
`model.safetensors` sha256 `c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1` is recorded in
`qwen3vl-embed-golden.json`), driving that repo's own `scripts/qwen3_vl_embedding.py` exactly as its README does. The
app's 4-bit checkpoint is **not** used here: the official bf16 model is the reference.

**Never edit these to make a change pass.** A mismatch is a `SPEC_QUESTIONS.md` entry, a bug in
`Modules/Embedding/Qwen3VLEmbed/`, or — if the recipe itself changed — a deliberate new fixture
regeneration with the owner's say-so.

- `vectors/qwen3vl_<name>.f32`: one 2048-value float32 little-endian vector per case, L2-normalised. The
  reference normalises in bf16 on MPS, so its stored norms are within ~0.2% of 1; the **app's** vectors
  are normalised in float32 and are the ones held to 1 ± 1e-4.
  - `card_query_1…4` and `card_text_document`: the model card's "Using transformers" example.
  - `sample_image`, `sample_image_matching_caption`, `sample_image_unrelated_caption`: the app sample
    image (`MLXUI/Resources/Samples/sample-image.png`, a peach card reading "Trailhead, 7am") alone,
    with a matching caption, and with an unrelated one.
  - `short_english_1`, `short_english_2`, `short_chinese`: three short texts of ours, one non-English.
- `qwen3vl-embed-golden.json`: revision, weights hash, package versions (`transformers` 5.22.2, `torch` 2.15.0,
  `qwen-vl-utils` 0.0.14, Python 3.13.3, device `mps:0`), the case table, the card's 4 × text-document
  similarity column this runner reproduces (0.8185 / 0.5185 / 0.3860 / 0.1083; the card prints
  0.8160 / 0.5173 / 0.3863 / 0.1061), and one rendered prompt + token-id capture for a text input and
  for an image input (the image prompt renders to 972 tokens, 950 of them `<|image_pad|>`, grid
  `[[1, 50, 76]]`).

Regenerate (a throwaway environment; the venv and both checkpoints are never committed):

```sh
hf download Qwen/Qwen3-VL-Embedding-2B \
  --revision 9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda \
  --local-dir /private/tmp/qwen3vl-embed-ref

python3 -m venv /private/tmp/qwen3vl-venv
/private/tmp/qwen3vl-venv/bin/pip install torch transformers qwen-vl-utils torchvision pillow numpy
/private/tmp/qwen3vl-venv/bin/python -I make_qwen3vl_embed_golden.py
```

The script defaults to the reference model at `/private/tmp/qwen3vl-embed-ref`, writes next to itself,
and runs the bf16 checkpoint on MPS when available (`--device cpu` forces the slow CPU path). It prints
the per-case norms and the card similarity column to stderr.
