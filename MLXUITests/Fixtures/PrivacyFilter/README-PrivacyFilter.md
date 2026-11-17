# OpenAI Privacy Filter reference fixtures (PF-3)

Generated **once** (2026-11-18) by `make_privacy_golden.py` — the committed generator in this folder.
**Never edit these to make a change pass**: a mismatch is a `SPEC_QUESTIONS.md` entry, or a bug in
`MLXUI/Modules/PrivacyFilter/`.

Two Python references produce them:

- **logits / hidden states / embeddings** — `transformers` 5.12.0
  `OpenAIPrivacyFilterForTokenClassification`, loaded in the dtype the checkpoint ships (bfloat16,
  **not** upcast), `eval()`, device `cpu`, `attn_implementation="eager"`, on the HF-layout checkpoint
  of `openai/privacy-filter` at revision `7ffa9a043d54d1be65afb281eddf0ffbe629385b`.
- **spans and redacted text** (the acceptance target) — `opf` at commit
  `f7f00ca7fb869683eb732c010299d901457f19c3`, `OPF(model=…, device="cpu",
  context_window_length=128000, trim_whitespace=True, discard_overlapping_predicted_spans=False,
  output_mode="typed", decode_mode="viterbi")`, using the checkpoint's default
  `viterbi_calibration.json` (all transition biases are `0.0`), on the opf original-layout checkpoint
  (the repo's `original/` subtree promoted to the directory root exactly as
  `opf/_common/checkpoint_download.py::_promote_original_subtree` does).

Recorded versions: **transformers 5.12.0, torch 2.7.0, numpy 2.2.6, opf 0.1.0 at commit
`f7f00ca7fb869683eb732c010299d901457f19c3`, huggingface_hub 1.19.0, tiktoken 0.14.0, Python
3.13.3**; torch device `cpu`, torch threads 6 (all recorded in `privacy-golden.json` under
`reference`).

The corpus is **100% synthetic** (decision 10): `example.com` addresses, `555` phone numbers,
invented names, an invalid `ZZ` state, `ACCT-…` account numbers and `sk-test-…` tokens. Fixture text
lives only in the fixture files — the generator prints case names, token counts, span counts and
byte lengths, never text.

## Reproduce

```sh
# Reference environment: a throwaway venv that reuses the system torch/transformers.
python3 -m venv --system-site-packages /tmp/pf-venv
/tmp/pf-venv/bin/pip install -e /tmp/opf-src --no-deps        # opf @ f7f00ca
/tmp/pf-venv/bin/pip install tiktoken huggingface_hub          # if not already present

# Checkpoints at HF revision 7ffa9a043d54d1be65afb281eddf0ffbe629385b:
#   /tmp/pf-checkpoint-hf   — HF/transformers layout: config.json, model.safetensors,
#                             tokenizer.json, tokenizer_config.json, viterbi_calibration.json
#   /tmp/pf-opf-checkpoint  — opf original layout: config.json, model.safetensors, dtypes.json,
#                             viterbi_calibration.json  (promoted from the repo's original/)
/tmp/pf-venv/bin/python -I make_privacy_golden.py \
    /tmp/pf-checkpoint-hf /tmp/pf-opf-checkpoint .
```

Measured final run (2026-11-18, this 6-thread host while other Xcode builds were running):
**253.9 s total** — HF model load 0.5 s; all transformers forwards 180.6 s (the 6,019-token long
document accounts for ~168 s of that; the 25 short texts are mostly <0.2 s each); probe trace 1.7 s;
opf load 5.5 s; opf prediction + cross-check 58.8 s (the long document's opf forward 53.6 s);
both weights sha256 2.9 s. The long document dominates; a machine-load-free run is faster.

`-I` (isolated mode) is the discipline the issue asks for: no user site, no `PYTHONPATH`, no current
directory on `sys.path`. The script is deterministic (fixed corpus, CPU inference, no randomness;
the wall-clock fields are informational only). `--shorts-only` writes the short corpus without the
long document (used to unblock PF-4/PF-5 early); the committed run is the full one.

The gated real-weights Swift tests read an installed copy at **`/private/tmp/pf-model`** — the five
HF-layout files above (`cp -c` clone from `/tmp/pf-checkpoint-hf`, sizes recorded below).

## Files

All `.f32` files are **raw little-endian float32** (`numpy` `<f4`), row-major, no header — a Swift
reader does `Data(contentsOf:)` + `bindMemory(to: Float.self)` and reshapes with the recorded shape.
All tensors have the batch dimension squeezed (`[T, C]`, not `[1, T, C]`).

| file | what |
|---|---|
| `make_privacy_golden.py` | the committed, deterministic generator (never run by tests) |
| `privacy-golden.json` | versions/revisions, conventions, the full corpus (text, token ids, per-token UTF-8 byte offsets, spans, `opf`'s redacted text), decoder logits references, the long document, and the measured cross-check |
| `parity.json` | PF-5's network-parity probe: one synthetic short text + token ids + tensor references + the exact load method |
| `parity-embedding.f32` | `[86, 640]` — the probe text's embedding output (`hidden_states[0]` == `embed_tokens(ids)`) |
| `parity-block-{0..7}.f32` | `[86, 640]` each — output of encoder layer `k` (post-MLP residual, **pre** final norm) |
| `parity-final-norm.f32` | `[86, 640]` — `model.model(ids).last_hidden_state` == `model.model.norm(parity-block-7)` (asserted bitwise) |
| `parity-logits.f32` | `[86, 33]` — the reference logits for the probe text |
| `pf-short-<case>.logits.f32` | `[T, 33]` per short case — reference pre-softmax logits |
| `pf-short-<case>.logprobs.f32` | `[T, 33]` per short case — `torch F.log_softmax(logits.float(), -1)`, the exact decoder input |
| `privacy-long-logits.f32` | `[LONG_TOKENS, 33]` — the long document's reference logits |
| `privacy-long-logprobs.f32` | `[LONG_TOKENS, 33]` — the long document's log-softmax |
| `privacy-repo-files.json` | the HF repo's file listing at the pinned revision (PF-2's pinned file list; Trap 9). It includes the ~12 GB `onnx/*` tree and the `original/*` subtree that the app must **not** download. |

`empty_string`'s two `.f32` files are 0 bytes (shape `[0, 33]`, no tokens). That is deliberate, not a
truncated file.

### File names: `privacy-golden.json`, not `golden.json`

The test target uses an Xcode `PBXFileSystemSynchronizedRootGroup`, which flattens every non-Swift
fixture file into a single resource namespace. `Fixtures/RIFE/golden.json`, `Fixtures/RIFE/README.md`
and `Fixtures/DepthAnything/repo-files.json` already exist, so this folder ships
`privacy-golden.json`, `README-PrivacyFilter.md` and `privacy-repo-files.json` (Depth Anything made
the same call, documented in `README-DepthAnything.md`). Swift reads the files through
`#filePath`, not the bundle, but a duplicate resource basename fails the build.

## Corpus

25 short texts plus one ~6k-token document, all built by the script (no text is hand-typed twice):

| case | covers |
|---|---|
| `person_alone`, `email_alone`, `phone_alone`, `url_alone`, `address_alone`, `date_alone`, `account_alone`, `secret_alone` | each of the eight categories alone |
| `mixed_all_categories` | all eight in one paragraph — also the PF-5 parity probe |
| `mixed_people_and_emails`, `mixed_phone_and_url`, `mixed_address_and_date`, `mixed_account_and_secret`, `names_with_punctuation` | categories mixed in pairs, punctuation/apostrophe/hyphen in a name |
| `negative_nonpersonal_date`, `negative_company_name`, `negative_example_url_in_prose` | negatives: a date, a company name, and an `example.com` URL in prose |
| `unicode_emoji_zwj`, `unicode_cjk`, `unicode_combining_accents`, `unicode_crlf`, `unicode_tabs`, `unicode_whitespace_edges` | Trap 1's Unicode/whitespace cases (emoji incl. ZWJ sequences, CJK, combining accents, CRLF, tabs, leading/trailing whitespace) |
| `whitespace_only`, `empty_string` | empty-after-trim inputs (Trap 1) |
| `long_document` | ~6,000 tokens built from repeated synthetic paragraphs with rotated people, emails, phones, URLs, addresses, dates, accounts and tokens |

Per short case in `privacy-golden.json`: `text`, `token_ids`, `token_byte_starts`,
`token_byte_ends`, `token_count`, `spans` (`label`, `start_byte`, `end_byte`, `text`,
`placeholder`), `redacted_text`, `logits`, `log_probs`, `decoded_mismatch` (always `false` — opf's
tokenizer round-trip is byte-exact on every text), and per-case `agreement`.

## Conventions

- **Token ids** are `tiktoken` `o200k_base`, `encode(text, allowed_special="all")`, no special
  tokens added. The HF `tokenizer.json` (i.e. what `HFTokenizerLoader` loads) produces identical
  ids — checked over the whole corpus and recorded in `tokenizer.token_ids_equal`.
- **Token byte offsets** are the prefix sums of `decode_single_token_bytes(token_id)`; the
  concatenated pieces equal the input's exact UTF-8 bytes. `token_byte_starts[i]`/`token_byte_ends[i]`
  are therefore the token's span in the input string's UTF-8, not in a normalized or re-decoded form.
- **Span offsets are UTF-8 byte offsets** into the exact JSON string's UTF-8 encoding, 0-based,
  end-exclusive. (opf works in Python character offsets; the generator converts each character
  boundary to its byte offset, and stores the character offsets only in memory for the cross-check.)
- **Span labels are opf's base labels**, never BIOES: one of `private_person`, `private_email`,
  `private_phone`, `private_url`, `private_address`, `private_date`, `account_number`, `secret`.
- **Spans are post-processed exactly as opf does it**: `trim_whitespace=True` (trim leading/trailing
  whitespace of each detected character span), then `_select_non_overlapping_spans` (sort by
  `(start, -(end - start), label)`, greedy keep non-overlapping, result sorted by start).
  `discard_overlapping_predicted_spans=False`.
- **`redacted_text`** is opf's `typed`-mode redaction: each selected span replaced by its
  placeholder (`<PRIVATE_PERSON>`, `<ACCOUNT_NUMBER>`, …), the rest of the input preserved
  byte-for-byte (including CRLF, tabs and whitespace).
- **`logits` vs `log_probs`**: `logits` are the reference pre-softmax outputs; `log_probs` is the
  exact tensor `opf`'s decoder received (`F.log_softmax(logits.float(), -1)`). A decoder-only test
  should use `log_probs` for a bit-exact comparison; they are `[T, 33]` float32.
- **`n_ctx`**: the generator passes `context_window_length=128000` to opf. opf hard-codes
  `n_ctx = 4096` when `device="cpu"` (`opf/_core/runtime.py::_resolve_n_ctx`), which would silently
  window the long document into 4,096-token, halo-less chunks — not a full-mask reference. 128,000 is
  the checkpoint's own `default_n_ctx` and what opf uses on CUDA, so this is opf's intended context
  length, not a deviation. It is unobservable for every short text (each is a single window under
  both values — verified structurally and recorded as
  `agreement.short_texts_single_window_at_4096_and_128000: true`). **Implementer's call, pending
  owner confirmation.**
- **Determinism**: the corpus, token ids, byte offsets, spans, redacted text, logits and
  `parity-*.f32` tensors are deterministic (fixed corpus, CPU inference, no randomness). The
  **informational** fields are not: `reference.generated` (generation date), `phase_seconds`,
  `total_seconds`, and each case's `opf_seconds`. A re-run reproduces every fixture tensor
  byte-for-byte and every span/redaction exactly; only those timestamps differ.
- **The long document is a full-mask reference**: `[~6000, 33]` logits from one full-sequence
  forward with the model's bidirectional `|i − j| ≤ 128` mask — matching the Swift `trace`/full-mask
  path PF-5 implements. The Swift 128,000-token windowed path is expected to reproduce it, not vice
  versa.

## Verification (measured, all recorded in `privacy-golden.json`)

- **Tokenizer-id equality** — `tiktoken o200k_base` vs HF `tokenizer.json`
  (`AutoTokenizer`, `add_special_tokens=False`): identical ids on **all 26 texts**
  (`tokenizer.token_ids_equal: true`, `mismatched_cases: []`); every text also round-trips
  byte-exactly (`decoded_mismatch: false`).
- **transformers-derived spans vs opf** — the stored transformers logits were pushed through opf's
  own `ViterbiCRFDecoder` and span post-processing (`predict_text` verbatim) and compared with
  `OPF.redact` on the same text: **26/26 texts agree exactly** — `agreement.mismatches: []`,
  `transformers_derived_spans_equal_opf: true`, and the derived redacted text equals `opf`'s on all
  26 (`transformers_derived_redacted_equals_opf: true`). The comparison is by
  `(label, character start, character end)` after opf's own span selection, then by redacted-text
  equality; it is not a tolerance.
- **Total spans in the opf reference** — 467 across the corpus: 42 in the 25 short texts and 425 in
  the 6,019-token long document. By label: `private_person` 120, `private_email` 56, `private_phone`
  52, `private_date` 70, `private_address` 48, `account_number` 48, `secret` 48, `private_url` 25.
  Every one of the eight category-alone texts yields exactly one span of its own category, the eight
  per-category labels all occur in the short corpus, and the three negatives / `whitespace_only` /
  `empty_string` yield none.
- **Determinism, measured** — two independent full runs (the first complete run and the committed
  one) produced **byte-identical `privacy-long-logits.f32`, `privacy-long-logprobs.f32` and all
  eleven `parity-*.f32` files** (sha256 compared); only the informational wall-clock fields and the
  `generated` date differ.
- **Independent artifact validation** (no model, run outside the generator): every case's
  `text_sha256`/`text_utf8_bytes` matches its text; tiktoken re-encoding reproduces `token_ids`;
  concatenating `decode_single_token_bytes` reproduces the text's UTF-8 exactly and matches
  `token_byte_starts/ends`; every span's bytes decode to its `text`, spans are non-overlapping and
  in-bounds; replacing spans by their `placeholder` reconstructs `redacted_text` byte-for-byte;
  every `.f32` file's byte size matches its recorded shape and every value is finite; the `files`
  inventory matches the folder. 0 errors.*

## Implementer's calls, pending owner confirmation

1. **`n_ctx = 128000` override** (above) — see `conventions.n_ctx_override_reason`.
2. **File names** `privacy-golden.json` / `README-PrivacyFilter.md` / `privacy-repo-files.json`
   instead of the issue's `golden.json` / `README.md` / `repo-files.json`, because of the flattened
   resource namespace (see above).
3. **Logits are stored for every short case**, not just the one probe text (+ its trace) and the long
   document. The issue's PF-6 asks for "the decoder alone, on the fixture logits, [to give] the
   fixture spans" on every corpus text; per-case logits + log_probs are ~1–6 KB each and make that
   test independent of the real weights. The full probe trace (`parity-*.f32`) is the one text the
   issue asks for embeddings/blocks/logits on, and it is also PF-5's parity fixture (`parity.json`).
4. **`blocks[k]` are captured with forward hooks, not `output_hidden_states`.** On this class
   `output_hidden_states=True` returns `num_hidden_layers + 1` entries where the last one is the
   **post-final-norm** `last_hidden_state` (it replaces the last layer's pre-norm output). So
   `blocks[0..7]` are the hook outputs of `model.model.layers[0..7]`, `parity-final-norm.f32` is
   `last_hidden_state`, and the generator asserts bitwise equality of
   `norm(blocks[7]) == last_hidden_state`. `parity.json` states this (`blocksArePreFinalNorm: true`,
   `finalNormIsLastHiddenState: true`).
5. **Sinks are used raw.** transformers adds the checkpoint's `self_attn.sinks` values directly as one
   extra attention-logit column before a fp32 softmax, with no `log(2)` scaling; opf's original code
   applies `S * ln(2)`, and the HF checkpoint's sinks are the opf values times `ln(2)` (verified
   numerically), so the two agree. `parity.json` records layer 0's 14 raw values.
6. **Logits are stored as float32** (the model runs bf16) so a Swift reader can compare without a
   bf16 unpacking path; the cast is `tensor.float()`.
7. **The long document's logits are stored in full** — the 65 fixture data files total
   3,949,048 B (`long_logits_storage.folder_bytes`), and the whole folder including this README,
   the generator and `privacy-golden.json` is ~4.4 MB — well under the issue's ~8 MB guidance, so
   the "checksum + every 64th row" fallback is not used
   (`long_logits_storage.fallback_used: false`). Of that, the long document's logits and log_probs
   are 794,508 B each (`[6019, 33]`).
8. **`empty_string` has 0-byte `.f32` files** rather than omitted keys, so a test can iterate every
   case uniformly.

## Reference versions, recorded

| item | value |
|---|---|
| HF repository / revision | `openai/privacy-filter` @ `7ffa9a043d54d1be65afb281eddf0ffbe629385b` |
| HF `model.safetensors` | 2,798,989,498 B, sha256 `06f66b87650b988b04e218285f9fe3df6a4943416b6ffa8171f07bc56cf12a9d` |
| opf `original/model.safetensors` | 2,798,984,088 B, sha256 `9c262cbe68a0c8a50590a648ef8341a2b7d3be1fa11dfb79893fe0b03ce57b5c` |
| opf source commit | `f7f00ca7fb869683eb732c010299d901457f19c3` |
| opf package version | 0.1.0 (editable install of that commit, `pip install -e /tmp/opf-src --no-deps`) |
| transformers / torch / numpy | 5.12.0 / 2.7.0 / 2.2.6 |
| huggingface_hub / tiktoken / Python | 1.19.0 / 0.14.0 / 3.13.3 |
| HF-layout checkpoint files (also `/private/tmp/pf-model`) | `config.json` 3,039 B; `model.safetensors` 2,798,989,498 B; `tokenizer.json` 27,868,174 B; `tokenizer_config.json` 234 B; `viterbi_calibration.json` 372 B |
| opf original-layout checkpoint files | `config.json` 707 B; `dtypes.json` 4,108 B; `model.safetensors` 2,798,984,088 B; `viterbi_calibration.json` 372 B |
