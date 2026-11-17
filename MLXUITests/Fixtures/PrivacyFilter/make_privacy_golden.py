"""One-time generator for MLXUITests/Fixtures/PrivacyFilter (issue PF-3).

Runs the two Python references on a fully synthetic corpus and writes the checked-in fixtures:

* **transformers 5.12.0** `OpenAIPrivacyFilterForTokenClassification`, loaded in the checkpoint's own
  **bfloat16**, `eval()`, device `cpu`, `attn_implementation="eager"` (the class has
  `_supports_sdpa = False`, so eager is the implemented path). It supplies the reference **logits**
  (and, for the one probe text, the embedding / per-block hidden states / final logits as `.f32`).
* **opf @ f7f00ca** (`OPF(model=..., device="cpu")`, `trim_whitespace=True`,
  `discard_overlapping_predicted_spans=False`, `output_mode="typed"`, `decode_mode="viterbi"`,
  default `viterbi_calibration.json`, whose biases are all zero). It supplies the reference
  **spans** and **redacted text** — the acceptance target.

and then **verifies** them against each other: the transformers logits are pushed through opf's own
Viterbi decoder and span post-processing (`opf/_core/runtime.py::predict_text` verbatim, on the HF
logits), and that result must equal what `OPF.redact` produced on the same text. Any disagreement is
recorded in `privacy-golden.json` under `agreement.mismatches` and printed as a count; nothing is
loosened.

Usage (never run from a committed environment; ordinary `python` needs torch + transformers + opf):

    /tmp/pf-venv/bin/python -I make_privacy_golden.py <hf_checkpoint> <opf_checkpoint> <out_dir> \
        [--shorts-only] [--skip-repo-files]

`<hf_checkpoint>` is the transformers/HF layout (`config.json`, `model.safetensors`,
`tokenizer.json`, `tokenizer_config.json`, `viterbi_calibration.json`); `<opf_checkpoint>` is the
opf original layout (`config.json`, `model.safetensors`, `dtypes.json`, `viterbi_calibration.json`),
i.e. the HF repo's `original/` subtree promoted to the directory root exactly as
`opf/_common/checkpoint_download.py::_promote_original_subtree` does. Both are pinned to HF revision
7ffa9a043d54d1be65afb281eddf0ffbe629385b.

`--shorts-only` writes the short corpus (and the parity probe) and leaves `long_document` null; the
final, committed run is the full one, which overwrites every file it writes and fills the long
document in. Every run is deterministic: the corpus is a fixed Python list, the models run on CPU
with a fixed thread count, and no randomness is used.

Fixture folder budget: the full corpus is well under the issue's ~8 MB guidance, so the long
document's logits are stored in full (no checksum / every-64th-row fallback) — see README-PrivacyFilter.md.

**Never log fixture text.** This script prints case names, token counts, span counts and byte
lengths only; the input text lives exclusively in the fixture files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------------------------------
# Pinned reference environment (recorded in privacy-golden.json and README-PrivacyFilter.md)
# --------------------------------------------------------------------------------------------------

HF_REPOSITORY = "openai/privacy-filter"
HF_REVISION = "7ffa9a043d54d1be65afb281eddf0ffbe629385b"
OPF_COMMIT = "f7f00ca7fb869683eb732c010299d901457f19c3"
ENCODING_NAME = "o200k_base"
EOT_TOKEN_ID = 199999
REFERENCE_DEVICE = "cpu"
ATTN_IMPLEMENTATION = "eager"

#: `OPF(..., context_window_length=...)` for every text. opf hard-codes n_ctx=4096 when
#: `device="cpu"` (`opf/_core/runtime.py::_resolve_n_ctx`), which would silently window the ~6,000
#: token document into 4,096-token, halo-less chunks — not a full-mask reference. 128,000 is the
#: checkpoint's own `default_n_ctx` (and what opf uses on CUDA), so this is opf's intended context
#: length, not a deviation. It is unobservable for every short text: each is one window under both
#: values (verified structurally below and recorded in the golden).
N_CTX = 128_000

#: Target size of the long document in tiktoken tokens (issue PF-3: "about 6,000 tokens").
LONG_TARGET_TOKENS = 6_000

#: The short case whose tensors are also exported as `parity.json` for PF-5's network parity test.
PROBE_CASE = "mixed_all_categories"

#: Files that make up the HF/transformers layout of the installed checkpoint (PF-2's file pinning).
HF_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "viterbi_calibration.json",
)


# --------------------------------------------------------------------------------------------------
# The synthetic corpus. All personal-looking values are invented: `example.com` addresses, 555
# numbers, placeholder names, an invalid `ZZ` state, `ACCT-` numbers and `sk-test-` tokens.
# --------------------------------------------------------------------------------------------------

SHORT_CASES: tuple[tuple[str, str], ...] = (
    # --- the eight categories, alone -------------------------------------------------------------
    ("person_alone", "Contact Jordan Samplewood at the front desk tomorrow morning."),
    ("email_alone", "Please send the signed form to taylor.sample@example.com before Friday."),
    ("phone_alone", "My phone number is 555-0142."),
    (
        "url_alone",
        "Profile: https://example.com/profiles/jordan-samplewood?ref=email",
    ),
    ("address_alone", "Ship the parcel to 221B Sample Street, Springfield, ZZ 12345."),
    ("date_alone", "The intake interview is scheduled for 03/14/1987."),
    ("account_alone", "Transfer the balance to account ACCT-0000-1234-9876."),
    ("secret_alone", "The staging access token is sk-test-0123456789abcdef0123456789abcdef."),
    # --- mixed categories -------------------------------------------------------------------------
    (
        "mixed_all_categories",
        "Jordan Samplewood (jordan.sample@example.com, 555-0188) asked that the renewal dated "
        "11/02/2024 be sent to https://example.com/renewals/1 and shipped to 221B Sample Street, "
        "Springfield, ZZ 12345, charged to account ACCT-0000-1234-9876 with token "
        "sk-test-0123456789abcdef0123456789abcdef.",
    ),
    (
        "mixed_people_and_emails",
        "Taylor Samplewood and Morgan Example both wrote from taylor.sample@example.com and "
        "morgan.example@example.com this week.",
    ),
    (
        "mixed_phone_and_url",
        "Call 555-0142 or visit the profile at "
        "https://example.com/profiles/jordan-samplewood?ref=email.",
    ),
    (
        "mixed_address_and_date",
        "The lease starts 06/01/2026 at 221B Sample Street, Springfield, ZZ 12345.",
    ),
    (
        "mixed_account_and_secret",
        "Use account ACCT-0000-1234-9876 and the token sk-test-0123456789abcdef0123456789abcdef "
        "for staging.",
    ),
    # --- negatives: not personal data --------------------------------------------------------------
    (
        "negative_nonpersonal_date",
        "The fiscal quarter ended on 31 March 2026 and the ledger closes at the end of the month.",
    ),
    (
        "negative_company_name",
        "Acme Sampleworks Corporation reported a modest gain this quarter.",
    ),
    (
        "negative_example_url_in_prose",
        "See the published style guide at https://example.com/style for details.",
    ),
    # --- Trap 1: Unicode, whitespace and line endings ----------------------------------------------
    (
        "unicode_emoji_zwj",
        "Family lunch with Alex Samplewood \U0001f9d1\u200d\U0001f4bb\U0001f469\u200d\U0001f469"
        "\u200d\U0001f467 at the caf\u00e9 \u2014 email alex.sample@example.com or ring 555-0177 "
        "\U0001f642.",
    ),
    (
        "unicode_cjk",
        "\u9023\u7d61\u5148\u306f\u30b5\u30f3\u30d7\u30eb\u682a\u5f0f\u4f1a\u793e\u3001"
        "\u62c5\u5f53\u306f\u5c71\u7530\u592a\u90ce\u3001\u30e1\u30fc\u30eb\u306f "
        "yamada.sample@example.com\u3001\u96fb\u8a71\u306f 555-0199 \u3067\u3059\u3002",
    ),
    (
        "unicode_combining_accents",
        "Jose\u0301 Samplewood, jose.sample@example.com \u2014 combining accents on the first name.",
    ),
    (
        "unicode_crlf",
        "Line one: Jordan Samplewood\r\nLine two: jordan.sample@example.com\r\n"
        "Line three: 555-0101\r\n",
    ),
    (
        "unicode_tabs",
        "Name:\tJordan Samplewood\tEmail:\tjordan.sample@example.com\tPhone:\t555-0102",
    ),
    (
        "unicode_whitespace_edges",
        "   \n  jordan.sample@example.com  \n   ",
    ),
    ("whitespace_only", "   \n\t \r\n  "),
    ("empty_string", ""),
    # --- punctuation around a name, and an international-format phone ------------------------------
    (
        "names_with_punctuation",
        "D'Arcy Samplewood-Jones asked that +1 555 0143 be used for calls and "
        "d.arcy.sample@example.com for mail.",
    ),
)

#: Repeated paragraph templates for the ~6,000-token document. `{i}` is a deterministic counter,
#: `{d}` a deterministic day; every value is synthetic.
LONG_PARAGRAPHS: tuple[str, ...] = (
    "Ticket {i:04d}: Jordan Samplewood reported a scheduling change for unit {unit}; reach the "
    "tenant at jordan.sample{i:04d}@example.com or 555-01{i:02d} on or after 07/{d:02d}/2026.",
    "Record {i:04d}: shipment {unit} was addressed to 221B Sample Street, Springfield, ZZ 12345 "
    "and charged to account ACCT-{i:04d}-0000-9876; the drop-box access token is "
    "sk-test-{i:04d}abcdef0123456789abcdef.",
    "Note {i:04d}: Taylor Samplewood discussed the renewal at "
    "https://example.com/renewals/{i:04d} and asked that the signed copy reach "
    "taylor.sample{i:04d}@example.com by 12/{d:02d}/2026.",
    "Follow-up {i:04d}: Morgan Example confirms the callback number 555-01{i:02d} and the "
    "shipping label for 221B Sample Street, Springfield, ZZ 12345.",
    "Audit {i:04d}: the reviewer checked the ledger entry of 03/{d:02d}/2026 against account "
    "ACCT-{i:04d}-0000-9876 and the placeholder token "
    "sk-test-{i:04d}abcdef0123456789abcdef for the staging environment.",
    "Summary {i:04d}: nobody on the team is a real person; Alex Samplewood, Jordan Samplewood and "
    "Taylor Samplewood are invented names standing in for the redaction corpus.",
)


def build_corpus() -> list[tuple[str, str]]:
    """Return the ordered `(case name, text)` short corpus (deterministic, no randomness)."""
    return list(SHORT_CASES)


def build_long_document(encoding) -> str:
    """Build the ~6,000-token document from the repeated synthetic paragraph templates.

    Deterministic: the paragraph index decides both the template and every substituted value, and
    the loop stops at the first paragraph boundary at or past `LONG_TARGET_TOKENS` tokens.
    """
    paragraphs: list[str] = []
    index = 0
    while True:
        template = LONG_PARAGRAPHS[index % len(LONG_PARAGRAPHS)]
        paragraphs.append(
            template.format(
                i=index,
                unit=(index % 97) + 1,
                d=(index % 27) + 1,
            )
        )
        index += 1
        if len(encoding.encode("\n\n".join(paragraphs), allowed_special="all")) >= LONG_TARGET_TOKENS:
            return "\n\n".join(paragraphs)


# --------------------------------------------------------------------------------------------------
# Tokenization and UTF-8 byte offsets
# --------------------------------------------------------------------------------------------------


def tokenize_with_byte_offsets(encoding, text: str) -> dict[str, object]:
    """Tokenize with tiktoken and return token ids plus per-token UTF-8 byte offsets.

    The offsets are the prefix sums of `decode_single_token_bytes`, i.e. each token's exact span in
    the input string's UTF-8 encoding. The concatenation of the token pieces must equal the input's
    bytes exactly; that is what the Swift tokenizer's contract asserts, so a mismatch here is fatal.
    """
    ids = [int(token) for token in encoding.encode(text, allowed_special="all")]
    pieces = [encoding.decode_single_token_bytes(token) for token in ids]
    joined = b"".join(pieces)
    expected = text.encode("utf-8")
    if joined != expected:
        raise SystemExit(
            "tokenizer round-trip failed: token bytes are not the input's exact UTF-8 "
            f"(tokens={len(ids)}, bytes={len(joined)}, expected_bytes={len(expected)})"
        )
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    for piece in pieces:
        starts.append(cursor)
        cursor += len(piece)
        ends.append(cursor)
    return {
        "token_ids": ids,
        "token_byte_starts": starts,
        "token_byte_ends": ends,
    }


def hf_tokenizer_ids(tokenizer, text: str) -> list[int]:
    """HF `tokenizer.json` ids, no special tokens added (== tiktoken o200k_base)."""
    return [int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"]]


# --------------------------------------------------------------------------------------------------
# Raw little-endian float32 tensors
# --------------------------------------------------------------------------------------------------


def write_f32(out_dir: Path, name: str, array) -> list[int]:
    """Write one array as row-major little-endian float32 and return its shape."""
    values = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    values.astype("<f4").tofile(out_dir / name)
    return [int(size) for size in values.shape]


def sha256_file(path: Path) -> str:
    """sha256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------------------
# Reference: transformers logits + trace
# --------------------------------------------------------------------------------------------------


def load_hf_model(hf_checkpoint: Path):
    """Load the pinned checkpoint in the dtype it ships in (bfloat16), eager, on CPU."""
    from transformers import AutoModelForTokenClassification

    model = AutoModelForTokenClassification.from_pretrained(
        str(hf_checkpoint),
        dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    model.eval()
    model.to(REFERENCE_DEVICE)
    if model.config._attn_implementation != ATTN_IMPLEMENTATION:
        raise SystemExit(
            f"reference attention implementation is {model.config._attn_implementation!r}, "
            f"expected {ATTN_IMPLEMENTATION!r}"
        )
    first_parameter = next(model.parameters())
    if first_parameter.dtype != torch.bfloat16:
        raise SystemExit(f"reference model loaded as {first_parameter.dtype}, expected bfloat16")
    return model


def hf_logits(model, token_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Reference logits and log-softmax for one text: [T, 33] float32 each."""
    if not token_ids:
        return np.zeros((0, model.config.num_labels), dtype=np.float32), np.zeros(
            (0, model.config.num_labels), dtype=np.float32
        )
    inputs = torch.tensor([token_ids], dtype=torch.long)
    with torch.inference_mode():
        outputs = model(input_ids=inputs)
    logits = outputs.logits[0].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    return (
        logits.numpy().astype(np.float32),
        log_probs.numpy().astype(np.float32),
    )


def hf_trace(model, token_ids: list[int]) -> dict[str, np.ndarray]:
    """Embedding, per-block hidden states, final norm and logits for the probe text.

    `output_hidden_states=True` on this class records `num_hidden_layers` entries:
    `[embedding, layer_0, ..., layer_6, post_final_norm]` — the last layer's *pre-norm* output is
    not in that tuple (transformers appends the final-norm result in its place). So the eight
    per-block outputs are captured with forward hooks on `model.model.layers[i]`; `parity.json`
    states this explicitly (`blocksArePreFinalNorm`, `finalNormIsLastHiddenState`).
    """
    layers = model.model.layers
    captured: dict[int, torch.Tensor] = {}
    hooks = [
        layer.register_forward_hook(
            lambda _module, _args, output, index=index: captured.__setitem__(index, output)
        )
        for index, layer in enumerate(layers)
    ]
    inputs = torch.tensor([token_ids], dtype=torch.long)
    try:
        with torch.inference_mode():
            outputs = model(input_ids=inputs, output_hidden_states=True)
            base = model.model(input_ids=inputs)
    finally:
        for hook in hooks:
            hook.remove()

    hidden_states = outputs.hidden_states
    if len(hidden_states) != model.config.num_hidden_layers + 1:
        raise SystemExit(
            f"output_hidden_states has {len(hidden_states)} entries, expected "
            f"{model.config.num_hidden_layers + 1}"
        )
    if len(captured) != model.config.num_hidden_layers:
        raise SystemExit(f"captured {len(captured)} layer outputs, expected 8")

    embedding = hidden_states[0]
    blocks = [captured[index] for index in range(len(layers))]
    final_norm = base.last_hidden_state
    if not torch.equal(hidden_states[-1], final_norm):
        raise SystemExit("hidden_states[-1] is not the base model's last_hidden_state")
    if not torch.equal(model.model.norm(blocks[-1]), final_norm):
        raise SystemExit("model.norm(layer_7 output) is not the base model's last_hidden_state")
    if not torch.equal(embedding, model.model.embed_tokens(inputs)):
        raise SystemExit("hidden_states[0] is not embed_tokens(input_ids)")

    with torch.inference_mode():
        logits = outputs.logits[0].float()
    return {
        "embedding": embedding[0].float().numpy().astype(np.float32),
        "blocks": [block[0].float().numpy().astype(np.float32) for block in blocks],
        "final_norm": final_norm[0].float().numpy().astype(np.float32),
        "logits": logits.numpy().astype(np.float32),
    }


# --------------------------------------------------------------------------------------------------
# Reference: opf spans + redacted text, and the transformers-logits cross-check
# --------------------------------------------------------------------------------------------------


def load_opf(opf_checkpoint: Path):
    """Load the opf reference runtime with the defaults the fixture records."""
    from opf import OPF

    redactor = OPF(
        model=str(opf_checkpoint),
        device=REFERENCE_DEVICE,
        context_window_length=N_CTX,
        trim_whitespace=True,
        discard_overlapping_predicted_spans=False,
        output_mode="typed",
        decode_mode="viterbi",
    )
    runtime = redactor.get_runtime()
    if runtime.n_ctx != N_CTX:
        raise SystemExit(f"opf runtime n_ctx is {runtime.n_ctx}, expected {N_CTX}")
    return redactor, runtime


def opf_byte_spans(redaction, text: str) -> list[dict[str, object]]:
    """Convert opf's character-offset spans into UTF-8 byte-offset span records."""
    byte_offsets = char_to_byte_offsets(text)
    spans: list[dict[str, object]] = []
    for span in redaction.detected_spans:
        start_byte = byte_offsets[span.start]
        end_byte = byte_offsets[span.end]
        spans.append(
            {
                "label": span.label,
                "start_byte": start_byte,
                "end_byte": end_byte,
                "text": span.text,
                "placeholder": span.placeholder,
            }
        )
    return spans


def char_to_byte_offsets(text: str) -> list[int]:
    """UTF-8 byte offset of every character boundary (length len(text) + 1)."""
    offsets = [0]
    cursor = 0
    for character in text:
        cursor += len(character.encode("utf-8"))
        offsets.append(cursor)
    return offsets


def transformers_spans(
    *,
    runtime,
    decoder,
    token_ids: list[int],
    logits: np.ndarray,
    text: str,
) -> tuple[list[object], str]:
    """Re-run opf's `predict_text` post-processing on transformers logits.

    This is the verification half of PF-3: it uses only the HF logits and opf's decoding/spans
    rules (`ViterbiCRFDecoder`, `labels_to_spans`, `token_spans_to_char_spans`,
    `trim_char_spans_whitespace`, `_select_non_overlapping_spans`, `_label_placeholder`), exactly as
    `opf/_core/runtime.py::predict_text` does, and returns (detected spans, redacted text).
    """
    import torch as torch_module

    from opf._core.runtime import (
        DetectedSpan,
        _label_placeholder,
        _select_non_overlapping_spans,
    )
    from opf._core.spans import (
        decode_text_with_offsets,
        labels_to_spans,
        token_spans_to_char_spans,
        trim_char_spans_whitespace,
    )

    if not token_ids:
        return [], text

    score_tensor = torch_module.from_numpy(np.asarray(logits, dtype=np.float32))
    log_probs = torch_module.log_softmax(score_tensor, dim=-1)
    decoded_labels = decoder.decode(log_probs)
    predicted_labels_by_index = {
        token_index: int(label) for token_index, label in enumerate(decoded_labels)
    }
    predicted_token_spans = labels_to_spans(predicted_labels_by_index, runtime.label_info)

    decoded_text, char_starts, char_ends = decode_text_with_offsets(
        token_ids, runtime.encoding
    )
    decoded_mismatch = decoded_text != text
    source_text = decoded_text if decoded_mismatch else text
    predicted_char_spans = token_spans_to_char_spans(
        predicted_token_spans, char_starts, char_ends
    )
    if runtime.trim_span_whitespace:
        predicted_char_spans = trim_char_spans_whitespace(predicted_char_spans, source_text)
    if runtime.discard_overlapping_predicted_spans:
        raise SystemExit("generator expects discard_overlapping_predicted_spans=False")

    detected = []
    for label_idx, start, end in predicted_char_spans:
        if not (0 <= start < end <= len(source_text)):
            continue
        label = str(runtime.label_info.span_class_names[label_idx])
        detected.append(
            DetectedSpan(
                label=label,
                start=int(start),
                end=int(end),
                text=source_text[start:end],
                placeholder=_label_placeholder(label),
            )
        )
    display_spans = _select_non_overlapping_spans(detected)
    pieces: list[str] = []
    cursor = 0
    for span in display_spans:
        pieces.append(source_text[cursor : span.start])
        pieces.append(span.placeholder)
        cursor = span.end
    pieces.append(source_text[cursor:])
    return display_spans, "".join(pieces)


def spans_equal(reference, candidate) -> bool:
    """Compare two sequences of opf `DetectedSpan`s by label and character offsets."""
    if len(reference) != len(candidate):
        return False
    for left, right in zip(reference, candidate):
        if (str(left.label), int(left.start), int(left.end)) != (
            str(right.label),
            int(right.start),
            int(right.end),
        ):
            return False
    return True


def mismatch_record(case: str, reference, candidate) -> dict[str, object]:
    """Metadata-only difference record (case name, counts, first differing positions)."""
    record: dict[str, object] = {
        "case": case,
        "opf_span_count": len(reference),
        "transformers_span_count": len(candidate),
    }
    for index in range(min(len(reference), len(candidate))):
        left, right = reference[index], candidate[index]
        if (left.label, left.start, left.end) != (right.label, right.start, right.end):
            record["first_difference"] = {
                "index": index,
                "opf": {"label": str(left.label), "start": int(left.start), "end": int(left.end)},
                "transformers": {
                    "label": str(right.label),
                    "start": int(right.start),
                    "end": int(right.end),
                },
            }
            break
    return record


# --------------------------------------------------------------------------------------------------
# HF repository listing (pf-2's pinned file list / Trap 9's "do not download onnx/*")
# --------------------------------------------------------------------------------------------------


def repo_files_document() -> dict[str, object]:
    """The HF repo's file listing at the pinned revision, sizes included."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(HF_REPOSITORY, revision=HF_REVISION, files_metadata=True)
    files = [
        {"path": sibling.rfilename, "size": int(sibling.size)}
        for sibling in info.siblings
        if sibling.size is not None
    ]
    files.sort(key=lambda entry: entry["path"])
    return {
        "repository": HF_REPOSITORY,
        "revision": HF_REVISION,
        "fetched": date.today().isoformat(),
        "source": (
            "GET https://huggingface.co/api/models/"
            f"{HF_REPOSITORY}/tree/{HF_REVISION}?recursive=true&expand=true"
        ),
        "files": files,
    }


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hf_checkpoint", type=Path)
    parser.add_argument("opf_checkpoint", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--shorts-only",
        action="store_true",
        help="write the short corpus and parity probe only (long_document left null)",
    )
    parser.add_argument(
        "--skip-repo-files",
        action="store_true",
        help="do not re-fetch privacy-repo-files.json from the HF API",
    )
    args = parser.parse_args(argv)

    import tiktoken
    import transformers
    from importlib.metadata import version as package_version

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    thread_count = torch.get_num_threads()

    phase_seconds: dict[str, float] = {}
    started = time.time()
    print(
        f"[env] torch {torch.__version__} transformers {transformers.__version__} "
        f"numpy {np.__version__} opf-commit {OPF_COMMIT} threads {thread_count}",
        flush=True,
    )

    encoding = tiktoken.get_encoding(ENCODING_NAME)
    if int(encoding.eot_token) != EOT_TOKEN_ID:
        raise SystemExit(f"encoding eot_token is {encoding.eot_token}, expected {EOT_TOKEN_ID}")

    # ---- corpus + tokenizer equality ------------------------------------------------------------
    cases = build_corpus()
    long_text = None if args.shorts_only else build_long_document(encoding)
    all_texts = [(name, text) for name, text in cases]
    if long_text is not None:
        all_texts.append(("long_document", long_text))

    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(str(args.hf_checkpoint))
    tokenizer_mismatches: list[str] = []
    tokenized: dict[str, dict[str, object]] = {}
    for name, text in all_texts:
        tokenized[name] = tokenize_with_byte_offsets(encoding, text)
        if hf_tokenizer_ids(hf_tokenizer, text) != tokenized[name]["token_ids"]:
            tokenizer_mismatches.append(name)
    print(
        f"[tokenizer] texts {len(all_texts)} mismatches {len(tokenizer_mismatches)} "
        f"eot {EOT_TOKEN_ID}",
        flush=True,
    )

    # ---- reference logits from transformers -----------------------------------------------------
    phase = time.time()
    hf_model = load_hf_model(args.hf_checkpoint)
    phase_seconds["hf_load"] = round(time.time() - phase, 2)
    print(
        f"[hf] loaded bf16 eager device={REFERENCE_DEVICE} "
        f"layers {hf_model.config.num_hidden_layers} labels {hf_model.config.num_labels} "
        f"in {phase_seconds['hf_load']}s",
        flush=True,
    )

    reference_logits: dict[str, dict[str, object]] = {}
    phase = time.time()
    for name, _text in all_texts:
        ids = tokenized[name]["token_ids"]
        logits, log_probs = hf_logits(hf_model, ids)
        logits_name = f"pf-short-{name}.logits.f32" if name != "long_document" else "privacy-long-logits.f32"
        logprobs_name = (
            f"pf-short-{name}.logprobs.f32"
            if name != "long_document"
            else "privacy-long-logprobs.f32"
        )
        logits_shape = write_f32(out_dir, logits_name, logits)
        logprobs_shape = write_f32(out_dir, logprobs_name, log_probs)
        reference_logits[name] = {
            "logits": {"file": logits_name, "shape": logits_shape, "dtype": "float32-le"},
            "log_probs": {
                "file": logprobs_name,
                "shape": logprobs_shape,
                "dtype": "float32-le",
                "kind": "torch F.log_softmax(logits.float(), -1) — the exact decoder input",
            },
        }
        print(
            f"[hf] {name}: tokens {len(ids)} logits {logits_shape}",
            flush=True,
        )
    phase_seconds["hf_logits"] = round(time.time() - phase, 2)

    # ---- probe trace (embeddings + per-block hidden states + final logits) ----------------------
    phase = time.time()
    probe_name = PROBE_CASE
    probe_ids = tokenized[probe_name]["token_ids"]  # type: ignore[assignment]
    trace = hf_trace(hf_model, probe_ids)  # type: ignore[arg-type]
    embedding_shape = write_f32(out_dir, "parity-embedding.f32", trace["embedding"])
    block_entries = []
    for index, block in enumerate(trace["blocks"]):
        file_name = f"parity-block-{index}.f32"
        block_entries.append(
            {"file": file_name, "shape": write_f32(out_dir, file_name, block)}
        )
    final_norm_shape = write_f32(out_dir, "parity-final-norm.f32", trace["final_norm"])
    parity_logits_shape = write_f32(out_dir, "parity-logits.f32", trace["logits"])
    layer_zero_sinks = [float(value) for value in hf_model.model.layers[0].self_attn.sinks.float().tolist()]
    parity_method = (
        f"transformers {transformers.__version__} OpenAIPrivacyFilterForTokenClassification, "
        "model loaded bf16 (torch.bfloat16), device=cpu, eval(), attn_implementation=eager, "
        "input_ids=tokenIds (batch 1, no padding); embedding = outputs.hidden_states[0] == "
        "model.model.embed_tokens(ids); blocks[k] = forward-hook output of model.model.layers[k] "
        "(post-MLP residual, PRE final norm) for k=0..7; finalNorm = model.model(ids)."
        "last_hidden_state == model.model.norm(blocks[7]) (asserted bitwise); logits = "
        "model(ids).logits; every saved tensor = tensor.float().numpy().astype('<f4') little-endian "
        "row-major with the batch dimension squeezed"
    )
    parity_document = {
        "shortText": probe_lookup_text(cases, probe_name),
        "case": probe_name,
        "tokenIds": probe_ids,
        "reference": {
            "embedding": {"file": "parity-embedding.f32", "shape": embedding_shape},
            "blocks": block_entries,
            "finalNorm": {"file": "parity-final-norm.f32", "shape": final_norm_shape},
            "logits": {"file": "parity-logits.f32", "shape": parity_logits_shape},
        },
        "blocksArePreFinalNorm": True,
        "finalNormIsLastHiddenState": True,
        "layerCount": len(block_entries),
        "method": parity_method,
        "torchDevice": REFERENCE_DEVICE,
        "attnImplementation": ATTN_IMPLEMENTATION,
        "loadedDtype": "bfloat16",
        "tensorLayout": (
            "little-endian float32, row-major [T, C], batch dimension squeezed; T = len(tokenIds)"
        ),
        "sinks": {
            "note": (
                "transformers adds the checkpoint sinks RAW as one extra attention-logit column "
                "(eager_attention_forward: torch.cat([attn_weights, module.sinks.reshape(...)], "
                "-1), fp32 softmax, column dropped); no log(2) scaling. opf's original code uses "
                "block.N.attn.sinks * ln(2), and the HF checkpoint's model.layers.N.self_attn.sinks "
                "equal the opf values times ln(2) (verified), so the two implementations agree: use "
                "the HF sinks raw."
            ),
            "layer0RawValues": layer_zero_sinks,
            "scaledByLog2": False,
        },
    }
    (out_dir / "parity.json").write_text(
        json.dumps(parity_document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    phase_seconds["hf_trace"] = round(time.time() - phase, 2)
    print(
        f"[hf] probe {probe_name}: tokens {len(probe_ids)} blocks {len(block_entries)} "
        f"trace_s {phase_seconds['hf_trace']}",
        flush=True,
    )

    del hf_model
    import gc

    gc.collect()

    # ---- reference spans/redaction from opf, and the cross-check ---------------------------------
    phase = time.time()
    opf_redactor, opf_runtime = load_opf(args.opf_checkpoint)
    from opf._core.decoding import build_sequence_decoder

    decoder, viterbi_biases = build_sequence_decoder(
        decode_mode="viterbi",
        label_info=opf_runtime.label_info,
        viterbi_calibration_path=None,
        checkpoint_dir=str(args.opf_checkpoint),
    )
    if decoder is None:
        raise SystemExit("opf did not build a Viterbi decoder")
    phase_seconds["opf_load"] = round(time.time() - phase, 2)
    print(
        f"[opf] loaded device={REFERENCE_DEVICE} n_ctx {opf_runtime.n_ctx} "
        f"bidirectional {opf_runtime.bidirectional_context} in {phase_seconds['opf_load']}s "
        f"biases {sorted(viterbi_biases.items()) if viterbi_biases else None}",
        flush=True,
    )

    def short_texts_stay_single_window() -> bool:
        """Every short text is one window at n_ctx=4096 (the CPU default) and at n_ctx=128000."""
        from opf._core.sequence_labeling import TokenizedExample, example_to_windows

        for name, text in cases:
            ids = tokenized[name]["token_ids"]
            example = TokenizedExample(
                tokens=tuple(ids),
                labels=tuple(0 for _ in ids),
                example_id=name,
                text=text,
            )
            narrow = list(example_to_windows(example, 4096))
            wide = list(example_to_windows(example, N_CTX))
            if len(narrow) != 1 or len(wide) != 1 or narrow[0].tokens != wide[0].tokens:
                return False
        return True

    single_window_short = short_texts_stay_single_window()

    golden_cases: dict[str, dict[str, object]] = {}
    agreement_mismatches: list[dict[str, object]] = []
    redaction_mismatches: list[str] = []
    span_count_total = 0

    phase = time.time()
    for name, text in all_texts:
        token_info = tokenized[name]
        ids = token_info["token_ids"]
        token_started = time.time()
        redaction = opf_redactor.redact(text)
        reference_spans = list(redaction.detected_spans)
        candidate_spans, candidate_redacted = transformers_spans(
            runtime=opf_runtime,
            decoder=decoder,
            token_ids=ids,  # type: ignore[arg-type]
            logits=load_f32(
                out_dir,
                str(reference_logits[name]["logits"]["file"]),  # type: ignore[index]
                reference_logits[name]["logits"]["shape"],  # type: ignore[index]
            ),
            text=text,
        )
        spans_agree = spans_equal(reference_spans, candidate_spans)
        redaction_agrees = candidate_redacted == redaction.redacted_text
        if not spans_agree:
            agreement_mismatches.append(mismatch_record(name, reference_spans, candidate_spans))
        if not redaction_agrees:
            redaction_mismatches.append(name)
        if redaction.text != text:
            raise SystemExit(f"{name}: opf reported a tokenizer round-trip mismatch")
        if redaction.warning is not None:
            raise SystemExit(f"{name}: opf reported a warning despite the byte-exact round-trip")

        byte_offsets = char_to_byte_offsets(text)
        span_records = [
            {
                "label": span.label,
                "start_byte": byte_offsets[span.start],
                "end_byte": byte_offsets[span.end],
                "text": span.text,
                "placeholder": span.placeholder,
            }
            for span in reference_spans
        ]
        span_count_total += len(span_records)
        unit = "short" if name != "long_document" else "long"
        case_record: dict[str, object] = {
            "group": unit,
            "text": text,
            "text_utf8_bytes": len(text.encode("utf-8")),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "token_count": len(ids),
            "token_ids": ids,
            "token_byte_starts": token_info["token_byte_starts"],
            "token_byte_ends": token_info["token_byte_ends"],
            "spans": span_records,
            "redacted_text": redaction.redacted_text,
            "decoded_mismatch": False,
            "opf_seconds": round(time.time() - token_started, 2),
        }
        case_record.update(reference_logits[name])
        case_record["agreement"] = {
            "transformers_derived_spans_equal_opf": spans_agree,
            "transformers_derived_redacted_equals_opf": redaction_agrees,
        }
        golden_cases[name] = case_record
        print(
            f"[opf] {name}: tokens {len(ids)} spans {len(span_records)} "
            f"bytes {case_record['text_utf8_bytes']} agree {spans_agree}/{redaction_agrees} "
            f"{case_record['opf_seconds']}s",
            flush=True,
        )
    phase_seconds["opf_and_verify"] = round(time.time() - phase, 2)

    # ---- weights hashes -------------------------------------------------------------------------
    phase = time.time()
    hf_weights_sha256 = sha256_file(args.hf_checkpoint / "model.safetensors")
    opf_weights_sha256 = sha256_file(args.opf_checkpoint / "model.safetensors")
    phase_seconds["weights_sha256"] = round(time.time() - phase, 2)

    # ---- repository listing ---------------------------------------------------------------------
    if args.skip_repo_files and (out_dir / "privacy-repo-files.json").exists():
        repo_document = json.loads((out_dir / "privacy-repo-files.json").read_text(encoding="utf-8"))
    else:
        repo_document = repo_files_document()
        (out_dir / "privacy-repo-files.json").write_text(
            json.dumps(repo_document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )

    # ---- the golden file ------------------------------------------------------------------------
    long_case = golden_cases.pop("long_document", None) if long_text is not None else None
    # Data files only: this generator, the hand-written README and the golden file itself are not
    # part of the inventory (the README is edited by hand after the run, so its size would drift).
    inventory_excluded = {"privacy-golden.json", "README-PrivacyFilter.md", "make_privacy_golden.py"}
    sizes = sorted(
        (entry.name, entry.stat().st_size)
        for entry in out_dir.iterdir()
        if entry.is_file() and entry.name not in inventory_excluded
    )
    total_bytes = sum(size for _name, size in sizes)
    golden = {
        "format_version": 1,
        "complete": long_case is not None,
        "generator": "make_privacy_golden.py",
        "reference": {
            "hf_repository": HF_REPOSITORY,
            "hf_revision": HF_REVISION,
            "hf_weights_sha256": hf_weights_sha256,
            "opf_commit": OPF_COMMIT,
            "opf_weights_sha256": opf_weights_sha256,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "opf": package_version("opf"),
            "huggingface_hub": package_version("huggingface_hub"),
            "tiktoken": package_version("tiktoken"),
            "python": sys.version.split()[0],
            "generated": date.today().isoformat(),
            "torch_device": REFERENCE_DEVICE,
            "torch_threads": thread_count,
            "attn_implementation": ATTN_IMPLEMENTATION,
            "hf_checkpoint_files": list(HF_CHECKPOINT_FILES),
            "opf_checkpoint_files": [
                "config.json",
                "model.safetensors",
                "dtypes.json",
                "viterbi_calibration.json",
            ],
        },
        "tokenizer": {
            "encoding": ENCODING_NAME,
            "eot_token": EOT_TOKEN_ID,
            "hf_tokenizer": "tokenizer.json (HF layout) via AutoTokenizer, add_special_tokens=False",
            "token_ids_equal": not tokenizer_mismatches,
            "texts_checked": len(all_texts),
            "mismatched_cases": tokenizer_mismatches,
        },
        "conventions": {
            "byte_offsets": (
                "UTF-8 byte offsets, 0-based, end exclusive, into the exact JSON string's UTF-8 "
                "encoding"
            ),
            "token_byte_offsets": (
                "per-token prefix sums of tiktoken decode_single_token_bytes(token_id); the "
                "concatenated pieces equal the input's UTF-8 bytes exactly"
            ),
            "token_ids": (
                "tiktoken o200k_base, encode(text, allowed_special='all'); no special tokens added"
            ),
            "span_labels": (
                "opf base labels, never BIOES: one of O (never emitted), account_number, "
                "private_address, private_date, private_email, private_person, private_phone, "
                "private_url, secret"
            ),
            "span_selection": (
                "opf trim_whitespace=True then _select_non_overlapping_spans: sort by "
                "(start, -(end - start), label), greedy keep non-overlapping, sorted by start"
            ),
            "redaction": (
                "opf OPF(...device='cpu', context_window_length=128000, trim_whitespace=True, "
                "discard_overlapping_predicted_spans=False, output_mode='typed', "
                "decode_mode='viterbi') with the checkpoint's default viterbi_calibration.json "
                "(all-zero biases); placeholder is the upper-cased label in angle brackets"
            ),
            "n_ctx_override_reason": (
                "opf hard-codes n_ctx=4096 for device='cpu'; 128000 is the checkpoint's "
                "default_n_ctx and the value opf itself uses on CUDA. Unobservable for every short "
                "text (single window under both values, verified structurally) and required for "
                "the long document to be a full-mask reference instead of halo-less 4096-token "
                "windows."
            ),
            "float_files": (
                "raw little-endian float32 (numpy '<f4'), row-major [T, C], batch squeezed; no "
                "header, no byte order mark — Data(contentsOf:) + bindMemory(to: Float.self)"
            ),
            "decoder_logits": (
                "per case: 'logits' are the reference pre-softmax logits; 'log_probs' is the exact "
                "decoder input torch produced (F.log_softmax(logits.float(), -1)). Use log_probs "
                "for a bit-exact decoder check."
            ),
        },
        "corpus": {
            "short_case_count": len(cases),
            "short_case_names": [name for name, _text in cases],
            "long_document_tokens": None if long_case is None else long_case["token_count"],
            "long_document_target_tokens": LONG_TARGET_TOKENS,
        },
        "probe": {
            "case": probe_name,
            "tokens": len(probe_ids),
            "parity_json": "parity.json",
            "reference": parity_document["reference"],
            "note": (
                "same probe text as the mixed_all_categories case; parity.json is PF-5's network "
                "parity fixture (embeddings + 8 per-block hidden states + final norm + logits)"
            ),
        },
        "cases": golden_cases,
        "long_document": long_case,
        "agreement": {
            "texts_checked": len(all_texts),
            "transformers_derived_spans_equal_opf": not agreement_mismatches,
            "transformers_derived_redacted_equals_opf": not redaction_mismatches,
            "mismatches": agreement_mismatches,
            "redaction_mismatches": redaction_mismatches,
            "short_texts_single_window_at_4096_and_128000": single_window_short,
            "total_opf_spans": span_count_total,
            "method": (
                "the stored transformers logits were pushed through opf's own ViterbiCRFDecoder "
                "and span post-processing (predict_text verbatim); the result was compared with "
                "OPF.redact() on the same text, by (label, character start, character end) and by "
                "redacted text equality"
            ),
        },
        "long_logits_storage": {
            "full_logits_stored": True,
            "fallback_used": False,
            "note": (
                "the full fixture folder is well under the issue's ~8 MB guidance, so the long "
                "document's logits and log_probs are stored in full (no checksum / every-64th-row "
                "fallback)"
            ),
            "folder_bytes": total_bytes,
        },
        "phase_seconds": phase_seconds,
        "total_seconds": round(time.time() - started, 1),
        "files": [{"name": name, "bytes": size} for name, size in sizes],
    }
    (out_dir / "privacy-golden.json").write_text(
        json.dumps(golden, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )

    print(
        f"[done] complete={golden['complete']} cases={len(golden_cases)} "
        f"long_tokens={None if long_case is None else long_case['token_count']} "
        f"spans={span_count_total} mismatches={len(agreement_mismatches)} "
        f"redaction_mismatches={len(redaction_mismatches)} "
        f"tokenizer_equal={not tokenizer_mismatches} "
        f"folder_bytes={total_bytes} total_s={golden['total_seconds']}",
        flush=True,
    )
    print(f"[done] phase_seconds {json.dumps(phase_seconds, sort_keys=True)}", flush=True)
    print(f"[done] files {len(sizes)} ({sum(1 for _ in sizes)}) wrote {out_dir}", flush=True)
    if agreement_mismatches:
        print(
            "[FATAL] transformers-derived spans differ from opf; recorded in "
            "agreement.mismatches — report to the Lead, do not loosen anything",
            flush=True,
        )
        return 2
    if tokenizer_mismatches:
        print(
            f"[FATAL] HF tokenizer ids differ from tiktoken for {len(tokenizer_mismatches)} text(s)",
            flush=True,
        )
        return 3
    return 0


def probe_lookup_text(cases: list[tuple[str, str]], name: str) -> str:
    """The probe case's text (looked up once so the corpus has a single source of truth)."""
    for case_name, text in cases:
        if case_name == name:
            return text
    raise SystemExit(f"probe case {name!r} is not in the corpus")


def load_f32(out_dir: Path, name: str, shape) -> np.ndarray:
    """Read back a fixture `.f32` and assert its shape."""
    values = np.fromfile(out_dir / name, dtype="<f4")
    expected = 1
    for size in shape:
        expected *= int(size)
    if values.size != expected:
        raise SystemExit(f"{name}: read {values.size} floats, expected {expected}")
    return values.reshape([int(size) for size in shape])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
