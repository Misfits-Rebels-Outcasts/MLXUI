#!/usr/bin/env python3
"""MC4-3 — reference vectors for Qwen3-VL Embedding 2B (multimodal embedder).

Run ONCE, in a throwaway environment, against the **official bf16** checkpoint
`Qwen/Qwen3-VL-Embedding-2B` at the pinned revision — never the app's 4-bit build. The official
repo is the reference: it ships `scripts/qwen3_vl_embedding.py` (Apache-2.0, Qwen), which this
script imports and drives exactly as its README does. Its `process()` wraps every input as the
model card's conversation (system "Represent the user's input.", an image item before the text
item, `add_generation_prompt=True`), takes the last token's final-norm hidden state and
L2-normalises it. That is decision 3 in the issue.

Regenerate (only with the owner's say-so):

    hf download Qwen/Qwen3-VL-Embedding-2B \
      --revision 9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda \
      --local-dir /private/tmp/qwen3vl-embed-ref
    python3 -m venv /private/tmp/qwen3vl-venv
    /private/tmp/qwen3vl-venv/bin/pip install torch transformers qwen-vl-utils torchvision pillow numpy
    /private/tmp/qwen3vl-venv/bin/python -I make_qwen3vl_embed_golden.py

The model directory defaults to `$QWEN3VL_EMBED_REF_DIR` or `/private/tmp/qwen3vl-embed-ref`;
the output directory defaults to this script's own directory; `--model-dir` / `--out-dir` /
`--image` override them. The image defaults to the app's `MLXUI/Resources/Samples/sample-image.png`.

What it writes (never commit the model or the venv):
  vectors/<name>.f32   — 2048 float32 little-endian values, L2-normalised, one file per case
  golden.json          — revision, package versions, case table, the card 4 x text-document
                         similarity column, and one rendered prompt + token-id capture for a
                         text input and for an image input
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
REFERENCE_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_MODEL_DIR = "/private/tmp/qwen3vl-embed-ref"
DIMENSIONS = 2048

# The model card's "Using transformers" example (README.md, revision 9f2f7e7): four queries and
# three documents. Only the text document is in this fixture — images and image+text documents
# are the image-path cases below, where the app can build them from sample-image.png.
CARD_QUERIES = [
    "A woman playing with her dog on a beach at sunset.",
    "Pet owner training dog outdoors near water.",
    "Woman surfing on waves during a sunny day.",
    "City skyline view from a high-rise building at night.",
]
CARD_TEXT_DOCUMENT = (
    "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at "
    "sunset, as the dog offers its paw in a heartwarming display of companionship and trust."
)
# The card prints this column (query x the text document) under "Sentence Transformers".
CARD_PRINTED_SIMILARITIES = [0.8160, 0.5173, 0.3863, 0.1061]

# Implementer's call: one wrong caption to go with the right one, plus three short texts of our
# own (the third non-English). sample-image.png is a plain peach card reading "Trailhead, 7am".
MATCHING_CAPTION = 'A plain peach-coloured sign that reads "Trailhead, 7am".'
UNRELATED_CAPTION = "A red sports car parked outside a garage on a rainy afternoon."
OWN_TEXTS = {
    "short_english_1": "A red bicycle leans against a brick wall.",
    "short_english_2": "Quarterly revenue rose by twelve percent.",
    "short_chinese": "一只橘猫睡在窗台上。",
}


def load_reference_module(model_dir: Path):
    """Import the official repo's `scripts/qwen3_vl_embedding.py` (Apache-2.0, Qwen)."""
    script = model_dir / "scripts" / "qwen3_vl_embedding.py"
    if not script.is_file():
        raise SystemExit(
            f"missing the official reference script at {script} — did you download "
            f"{REFERENCE_MODEL} at {REVISION}?"
        )
    spec = importlib.util.spec_from_file_location("qwen3_vl_embedding", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def capture_tokens(embedder, image: Path, conversation):
    """One conversation through the reference's own preprocess, to pin prompt + token ids."""
    text = embedder.processor.apply_chat_template(
        [conversation], add_generation_prompt=True, tokenize=False
    )
    if isinstance(text, (list, tuple)):  # transformers 5 returns a batch of rendered strings
        text = text[0]
    inputs = embedder._preprocess_inputs([conversation])
    tokens = inputs["input_ids"][0].tolist()
    mask = inputs.get("attention_mask")
    capture = {
        "rendered_prompt": text,
        "token_ids": [int(t) for t in tokens],
        "token_count": len(tokens),
        "attention_mask": [int(m) for m in mask[0].tolist()] if mask is not None else None,
    }
    grid = inputs.get("image_grid_thw")
    if grid is not None:
        capture["image_grid_thw"] = [[int(v) for v in row] for row in grid.tolist()]
    if inputs.get("pixel_values") is not None:
        capture["pixel_values_shape"] = list(inputs["pixel_values"].shape)
    return capture


def main() -> None:
    import argparse

    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=__import__("os").environ.get("QWEN3VL_EMBED_REF_DIR", DEFAULT_MODEL_DIR),
        help="the official bf16 reference checkpoint directory",
    )
    parser.add_argument("--out-dir", default=str(here), help="where vectors/ and golden.json go")
    parser.add_argument("--device", default=None, choices=["mps", "cpu"], help="reference compute device")
    parser.add_argument(
        "--image",
        default=str(here.parent.parent.parent / "MLXUI" / "Resources" / "Samples" / "sample-image.png"),
        help="the app sample image used for the image cases",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    image = Path(args.image).resolve()
    if not image.is_file():
        raise SystemExit(f"missing image {image}")
    vectors_dir = out_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    ref = load_reference_module(model_dir)
    import importlib.metadata
    import qwen_vl_utils
    import torch
    import transformers

    embedder = ref.Qwen3VLEmbedder(model_name_or_path=str(model_dir))
    # The reference script hardcodes CPU. This runs the bf16 checkpoint on MPS when available:
    # a bf16 CPU forward over the vision tower takes tens of minutes (torch's CPU bf16 path is
    # scalar), MPS takes seconds, and the vector it produces is the same model. `--device cpu`
    # (or QWEN3VL_EMBED_DEVICE=cpu) forces the slow path.
    device_name = args.device or __import__("os").environ.get("QWEN3VL_EMBED_DEVICE", "mps")
    device = torch.device(device_name if device_name == "cpu" or torch.backends.mps.is_available() else "cpu")
    embedder.model.to(device)
    print(f"reference device: {embedder.model.device}", file=sys.stderr)

    cases = []
    for index, query in enumerate(CARD_QUERIES, start=1):
        cases.append({"name": f"card_query_{index}", "kind": "text", "text": query})
    cases.append({"name": "card_text_document", "kind": "text", "text": CARD_TEXT_DOCUMENT})
    cases.append({"name": "sample_image", "kind": "image", "image": str(image)})
    cases.append(
        {
            "name": "sample_image_matching_caption",
            "kind": "image_text",
            "image": str(image),
            "text": MATCHING_CAPTION,
        }
    )
    cases.append(
        {
            "name": "sample_image_unrelated_caption",
            "kind": "image_text",
            "image": str(image),
            "text": UNRELATED_CAPTION,
        }
    )
    for name, text in OWN_TEXTS.items():
        cases.append({"name": name, "kind": "text", "text": text})

    torch.set_printoptions(precision=6, sci_mode=False)
    table = []
    for case in cases:
        item = {}
        if "text" in case:
            item["text"] = case["text"]
        if "image" in case:
            item["image"] = case["image"]
        # One input at a time: a joint batch pads every row to the longest image prompt, which
        # multiplies the vision-token attention work for the text-only cases for no benefit.
        vector = embedder.process([item])[0]
        values = vector.detach().to(torch.float32).cpu().numpy()
        if values.shape != (DIMENSIONS,):
            raise SystemExit(f"{case['name']}: expected {DIMENSIONS} dims, got {values.shape}")
        path = vectors_dir / f"qwen3vl_{case['name']}.f32"
        path.write_bytes(struct.pack(f"<{DIMENSIONS}f", *[float(v) for v in values]))
        norm = float(sum(float(v) * float(v) for v in values) ** 0.5)
        row = dict(case)
        if "image" in row:
            row["image"] = "MLXUI/Resources/Samples/sample-image.png"
        row.update(
            {
                "file": f"vectors/{path.name}",
                "dimensions": DIMENSIONS,
                "norm": norm,
                "bytes": path.stat().st_size,
            }
        )
        table.append(row)
        print(f"{case['name']}: norm={norm:.6f} dim={len(values)}", file=sys.stderr)

    # The card's 4 x text-document similarity column, recomputed from the reference vectors.
    by_name = {row["name"]: row["file"] for row in table}
    doc = struct.unpack(
        f"<{DIMENSIONS}f", (out_dir / by_name["card_text_document"]).read_bytes()
    )
    similarities = []
    for index in range(1, len(CARD_QUERIES) + 1):
        query = struct.unpack(
            f"<{DIMENSIONS}f", (out_dir / by_name[f"card_query_{index}"]).read_bytes()
        )
        similarities.append(sum(a * b for a, b in zip(query, doc)))
    print("card 4 x text-document similarities:", similarities, file=sys.stderr)
    print("card printed column:            ", CARD_PRINTED_SIMILARITIES, file=sys.stderr)

    # Prompt + token captures for MC4-4/MC4-5: one text input, one image input.
    text_conversation = embedder.format_model_input(text=CARD_QUERIES[0])
    image_conversation = embedder.format_model_input(image=str(image))
    captures = {
        "text": capture_tokens(embedder, image, text_conversation),
        "image": capture_tokens(embedder, image, image_conversation),
    }

    golden = {
        "reference_model": REFERENCE_MODEL,
        "revision": REVISION,
        "golden_model_sha256": _sha256(model_dir / "model.safetensors"),
        "generator": "make_qwen3vl_embed_golden.py",
        "generated_on": "2026-11-07",
        "versions": {
            "python": sys.version.split()[0],
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "qwen_vl_utils": importlib.metadata.version("qwen-vl-utils"),
            "reference_script": "scripts/qwen3_vl_embedding.py from the model at the pinned revision",
            "device": str(embedder.model.device),
        },
        "recipe": {
            "system_prompt": "Represent the user's input.",
            "add_generation_prompt": True,
            "pooling": "last token of the final-norm hidden state (include_prompt=true)",
            "normalization": "L2, 2048 dimensions, no Matryoshka truncation",
        },
        "dimensions": DIMENSIONS,
        "cases": table,
        "card_text_document_similarities": similarities,
        "card_printed_similarities": CARD_PRINTED_SIMILARITIES,
        "captures": captures,
    }
    (out_dir / "qwen3vl-embed-golden.json").write_text(json.dumps(golden, indent=2) + "\n")
    print(f"wrote {out_dir / 'golden.json'}", file=sys.stderr)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
