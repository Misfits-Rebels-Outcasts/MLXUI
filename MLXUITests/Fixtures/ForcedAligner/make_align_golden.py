#!/usr/bin/env python3
"""Generate the FA-4 reference fixtures for Qwen3-ForcedAligner (issue #13).

WHAT THIS DOES
  1. Generates three short clips with macOS `say` (English, Chinese, Japanese),
     exactly the texts/voices/rate recorded in CLIPS below, as 16 kHz mono
     LEI16 WAV (committed as fixtures when the folder is under ~2 MB).
  2. Loads the pinned Python reference -- the `mlx-audio` package whose
     conversion produced the MLX weights -- from a LOCAL checkpoint directory.
  3. Runs forced alignment for each clip and writes `golden-ForcedAligner.json` (the
     reference's (text, start, end) items) plus this fixture folder's README.
     README-ForcedAligner.md records the HF repo id + revision and the mlx-audio version.

RUN IT ONCE, ISOLATED, FROM THE FIXTURE FOLDER

    python3 -I make_align_golden.py --model-dir /private/tmp/fa-model

`python3 -I` is required by FA-4. A side effect matters here: `-I` ignores the
user site-packages. On the host this was developed on, `cffi` is installed only
in the user site, so the package's WAV loader
(`mlx_audio.stt.utils.load_audio` -> `miniaudio` -> `_cffi_backend`) raises
`ModuleNotFoundError` under `-I`. This script therefore never touches that
loader: it reads the WAV with the stdlib `wave` module and passes a float32
numpy array straight into `generate()`, which the aligner accepts. The int16
samples are divided by 32768.0, which is what miniaudio does, so the model sees
the same waveform it would get from the path-based API. numpy is required
(already an mlx-audio dependency).

WHAT THE REFERENCE RETURNS
  `model.generate(audio, text, language)` returns `ForcedAlignResult` for a
  single input (`mlx_audio/stt/models/qwen3_asr/qwen3_forced_aligner.py:637`,
  `:760`). `result.items` is a list of `ForcedAlignItem(text, start_time,
  end_time)` (`:299`) in transcript order, in SECONDS, already rounded to three
  decimals by the reference (`:744-753`).

UNITS
  A unit is whatever the reference's `ForceAlignProcessor.encode_timestamp`
  makes (`:246`): a whitespace-separated word for English, one character per
  CJK ideograph for Chinese, a nagisa word for Japanese, a soynlp word for
  Korean. Characters that are not letters/numbers/apostrophe are dropped
  (`:23-34`), so punctuation never appears in `items`, and `result.text` joins
  units with spaces.

JAPANESE -- language="Japanese", THE NAME THE SWIFT ENGINE PASSES
  Owner ruling 1 (Boo, 2026-11-27): "Accept it only if it matches the language
  name the Swift engine passes. Regenerate the Japanese golden with the
  library's full name for Japanese. If the units and times are identical, keep
  it and note that in the fixture README. If they differ, the golden uses the
  same name Swift passes."

  `ForcedAlignerEngine.languageNames["ja"]` is "Japanese", and the two spellings
  give different units, so the golden uses "Japanese". Python then takes its
  nagisa path (`:256-266`, nagisa must be installed) and times こんにちは、世界。
  as nagisa words: ["こんにちは", "世界"]. The pinned Swift runner
  (mlx-audio-swift @ 5c4c12b,
  Sources/MLXAudioSTT/Models/Qwen3ASR/Qwen3ForcedAligner.swift,
  `encodeTimestamp`) has no Japanese tokenizer: every language but "chinese"
  goes to `tokenizeSpaceLang`, which keeps a kana run whole and makes each kanji
  its own unit: ["こんにちは", "世", "界"].

  Owner ruling 8 (Boo, 2026-11-27): "Option A. Use the 'Japanese' golden. In the
  parity test only, for Japanese only, join consecutive Swift units whose
  concatenated text exactly equals one reference unit, then apply ±80 ms to the
  joined span. If the join doesn't line up exactly, the test fails. The engine
  and the app's output stay as the library splits them (issue #13 decision 4:
  don't re-split). Add a test that pins Swift's current Japanese units
  (こんにちは, 世, 界), so a future upstream splitter shows up as a deliberate
  change. Note the difference in the fixture README, and add it to the same
  upstream mlx-audio-swift report as the Chinese drift (ruling 2). Chinese stays
  as ruling 2: exact units, ±160 ms."

  That upstream report (the mel drift, the Japanese/Korean splitter and the
  newline split) is https://github.com/Neywa-Labs/mlx-audio-swift/issues/422.

PROVENANCE
  `mlx-audio` 0.4.4 is installed from a wheel and ships no VCS metadata (no
  `direct_url.json`), so there is no commit to record. golden-ForcedAligner.json records the
  package version plus sha256 hashes of the reference source files, which are
  what actually determine the output; the README says the same. The model
  checkpoint's own README says it was converted with mlx-audio 0.3.1.

Deterministic: `argmax` decoding, no sampling; running this twice produces
byte-identical golden-ForcedAligner.json (apart from clip timing prints, which are not
stored). Do not edit golden-ForcedAligner.json by hand -- regenerate it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

HF_REPO_ID = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
HF_REVISION = "0e1a68e91d815300c7c9754b2a7639378b23db15"
HF_PINNED_DATE = "2026-11-19"
BASE_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
BASE_MODEL_LICENSE = "Apache-2.0"
# The committed file list of the pinned revision (next to this script, whatever --out-dir is).
REPO_FILES_NAME = "repo-files-ForcedAligner.json"
MODEL_CARD_CONVERSION_VERSION = "0.3.1"  # stated in the checkpoint's README.md

# mlx-audio 0.4.4 shipped as a PyPI wheel, which carries no VCS metadata, but
# all six reference files below hash byte-identically to the files at upstream
# tag v0.4.4 -- so that tag's commit is the exact code that produced the
# goldens. Verified 2026-11-19:
#   git ls-remote https://github.com/Blaizzy/mlx-audio.git 'refs/tags/v0.4.4*'
#   -> a7ef98604cfd752e9e5c9011bcee8ec8c67228be refs/tags/v0.4.4
#   raw.githubusercontent.com/Blaizzy/mlx-audio/v0.4.4/mlx_audio/<file>
#   sha256 == installed sha256 for every file in REFERENCE_FILES.
REFERENCE_TAG = "v0.4.4"
REFERENCE_TAG_COMMIT = "a7ef98604cfd752e9e5c9011bcee8ec8c67228be"

# The reference source files that determine the output (relative to the
# mlx_audio package root). Hashed into golden-ForcedAligner.json so a drifted install is
# visible even though the wheel carries no commit.
REFERENCE_FILES = [
    "stt/models/qwen3_asr/qwen3_forced_aligner.py",
    "stt/models/qwen3_asr/qwen3_asr.py",
    "stt/models/qwen3_asr/config.py",
    "stt/utils.py",
    "utils.py",
    "version.py",
]

SAMPLE_RATE = 16000

# One entry per committed clip. `language` is the FA contract code (decision 6);
# `referenceLanguage` is what is handed to the Python reference.
# `rate` is pinned so a regeneration on the same macOS voice matches; the WAV
# files themselves are committed, so the goldens are tied to the committed
# bytes and this only matters with --force-audio.
CLIPS = [
    {
        "file": "align-en.wav",
        "language": "en",
        "referenceLanguage": "English",
        "voice": "Samantha",
        "rate": 180,
        "text": "The quick brown fox jumps over the lazy dog.",
    },
    {
        "file": "align-zh.wav",
        "language": "zh",
        "referenceLanguage": "Chinese",
        "voice": "Tingting",
        "rate": 180,
        "text": "今天天气很好，我们去公园散步。",
    },
    {
        "file": "align-ja.wav",
        "language": "ja",
        # The name the Swift engine passes (owner ruling 1); see JAPANESE in the
        # module docstring.
        "referenceLanguage": "Japanese",
        "voice": "Kyoko",
        "rate": 180,
        "text": "こんにちは、世界。",
    },
]


# --------------------------------------------------------------------------- #
# clips
# --------------------------------------------------------------------------- #


def say_command(clip: dict, out_path: Path) -> list[str]:
    """Exact `say` invocation for one clip (also recorded in the README)."""
    return [
        "say",
        "-v",
        clip["voice"],
        "-r",
        str(clip["rate"]),
        "-o",
        str(out_path),
        "--data-format=LEI16@16000",
        clip["text"],
    ]


def generate_clip(clip: dict, out_dir: Path, force: bool) -> Path:
    """Generate one clip with `say` unless the committed WAV is already there."""
    path = out_dir / clip["file"]
    if path.exists() and not force:
        return path
    if shutil.which("say") is None:
        raise SystemExit("`say` not found: the clip generator must run on macOS")
    print(f"  say -> {path.name}", flush=True)
    subprocess.run(say_command(clip, path), check=True)
    return path


def read_wav(path: Path) -> np.ndarray:
    """Read a 16 kHz mono LEI16 WAV into the float32 samples the model expects.

    Deliberately avoids `mlx_audio.stt.utils.load_audio`: that goes through
    miniaudio/cffi, and `python3 -I` hides the user site where cffi lives on
    this host. Dividing by 32768.0 is what miniaudio's int16 -> float32
    conversion does.
    """
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise SystemExit(
                f"{path.name}: expected mono 16-bit PCM, got "
                f"{wav.getnchannels()} channel(s), {wav.getsampwidth()*8}-bit"
            )
        if wav.getframerate() != SAMPLE_RATE:
            raise SystemExit(
                f"{path.name}: expected {SAMPLE_RATE} Hz, got {wav.getframerate()} Hz"
            )
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def reference_provenance() -> dict:
    import mlx_audio

    package_root = Path(mlx_audio.__file__).resolve().parent
    hashes = {}
    for relative in REFERENCE_FILES:
        path = package_root / relative
        hashes[relative] = sha256_file(path) if path.is_file() else "missing"
    version = package_version("mlx-audio")
    # The installed wheel carries no VCS metadata (no direct_url.json,
    # INSTALLER=pip). The commit below is upstream tag v0.4.4, established by
    # matching the installed sources against that tag; only claim it for 0.4.4.
    commit = REFERENCE_TAG_COMMIT if version == "0.4.4" else None
    commit_note = (
        f"upstream tag {REFERENCE_TAG} (PyPI wheel {version}, installed by pip). "
        "The wheel ships no VCS metadata; this commit was established by "
        "sha256-comparing the installed reference sources against the files at "
        f"tag {REFERENCE_TAG} in github.com/Blaizzy/mlx-audio "
        "(all six match). The hashes below are what actually produced the "
        "goldens."
        if commit
        else "not recorded: this version was not verified against an upstream tag"
    )
    return {
        "mlxAudioVersion": version,
        "mlxAudioTag": REFERENCE_TAG if commit else None,
        "mlxAudioCommit": commit,
        "mlxAudioCommitNote": commit_note,
        "mlxAudioFileSha256": hashes,
        "modelCardConversionVersion": MODEL_CARD_CONVERSION_VERSION,
        "pythonVersion": platform.python_version(),
        "mlxVersion": package_version("mlx"),
        "transformersVersion": package_version("transformers"),
        "numpyVersion": package_version("numpy"),
        "nagisaVersion": package_version("nagisa"),  # the Japanese word splitter
        "platform": f"{platform.system()} {platform.machine()}",
    }


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #


def load_reference(model_dir: Path):
    from mlx_audio.stt.utils import load_model

    if not (model_dir / "config.json").is_file():
        raise SystemExit(f"{model_dir}: no config.json -- point --model-dir at the checkpoint folder")
    if not (model_dir / "model.safetensors").is_file():
        raise SystemExit(f"{model_dir}: no model.safetensors -- is the download complete?")

    started = time.time()
    # Local directory: never a repo id, so nothing can be fetched (Trap 1).
    model = load_model(str(model_dir))
    elapsed = time.time() - started

    # Sanity: the package must resolve this config to the forced aligner, not to
    # the Qwen3-ASR wrapper (`mlx_audio/stt/models/qwen3_asr/config.py:97-106`,
    # `qwen3_asr.py:1375-1393`).
    config_name = type(model.config).__name__
    inner = getattr(model, "_model", model)
    inner_name = type(inner).__name__
    if config_name != "ForcedAlignerConfig" or inner_name != "ForcedAlignerModel":
        raise SystemExit(
            "loaded the wrong model class: "
            f"config={config_name}, model={inner_name}; expected "
            "ForcedAlignerConfig/ForcedAlignerModel"
        )
    print(
        f"  loaded {inner_name} from {model_dir} in {elapsed:.2f}s "
        f"(timestamp_token_id={model.config.timestamp_token_id}, "
        f"segment={model.config.timestamp_segment_time} ms, "
        f"classes={model.config.classify_num})",
        flush=True,
    )
    return model


def align_clip(model, clip: dict, path: Path) -> dict:
    samples = read_wav(path)
    duration = len(samples) / SAMPLE_RATE

    started = time.time()
    result = model.generate(
        audio=samples,
        text=clip["text"],
        language=clip["referenceLanguage"],
    )
    elapsed = time.time() - started

    items = [
        {
            "text": item.text,
            "start": float(item.start_time),
            "end": float(item.end_time),
        }
        for item in result.items
    ]

    # The splitter is the contract (decision 4): items must be exactly the
    # units `encode_timestamp` asked the model to time, in order.
    word_list, _ = model.aligner_processor.encode_timestamp(
        clip["text"], clip["referenceLanguage"]
    )
    if [item["text"] for item in items] != word_list:
        raise SystemExit(
            f"{clip['file']}: items {[item['text'] for item in items]!r} do not "
            f"match the splitter {word_list!r}"
        )
    for item in items:
        if item["end"] < item["start"]:
            raise SystemExit(f"{clip['file']}: end < start in {item!r}")
    starts = [item["start"] for item in items]
    if starts != sorted(starts):
        raise SystemExit(f"{clip['file']}: starts are not monotonic: {starts!r}")

    print(
        f"  {clip['file']}: {len(items)} units, {duration:.2f}s audio, "
        f"aligned in {elapsed:.2f}s",
        flush=True,
    )
    for item in items:
        print(f"      {item['text']!r:20} {item['start']:7.3f}  {item['end']:7.3f}")

    return {
        "file": clip["file"],
        "language": clip["language"],
        "referenceLanguage": clip["referenceLanguage"],
        "voice": clip["voice"],
        "sayRate": clip["rate"],
        "text": clip["text"],
        "sampleRate": SAMPLE_RATE,
        "channels": 1,
        "duration": round(duration, 3),
        "fileSha256": sha256_file(path),
        "items": items,
    }


# --------------------------------------------------------------------------- #
# README
# --------------------------------------------------------------------------- #


def load_repo_files() -> dict:
    """The pinned revision's file list; refuses one that describes another pin."""
    repo_files = json.loads(
        (Path(__file__).resolve().parent / REPO_FILES_NAME).read_text(encoding="utf-8")
    )
    pin = (repo_files["repository"], repo_files["revision"])
    if pin != (HF_REPO_ID, HF_REVISION):
        raise SystemExit(f"{REPO_FILES_NAME} describes {pin}, not {(HF_REPO_ID, HF_REVISION)}")
    return repo_files


def pinned_checkpoint_section(golden: dict, repo_files: dict) -> list[str]:
    """The README's opening: the pinned checkpoint and the download selection.

    All of this text lives here and nowhere else; the table's values are data,
    the paragraph under it is prose (edit it here, then regenerate).
    """
    count = len(repo_files["files"])
    total = sum(entry["size"] for entry in repo_files["files"])
    return [
        "# ForcedAligner fixtures",
        "",
        "FA-2/FA-4 for GitHub issue #13 (`Qwen3-ForcedAligner 0.6B`).",
        "",
        "## Pinned checkpoint and download selection",
        "",
        "| | |",
        "|---|---|",
        f"| Repository | `{golden['hfRepoId']}` |",
        f"| Revision (pinned {HF_PINNED_DATE}) | `{golden['hfRevision']}` |",
        f"| Base model | `{BASE_MODEL}` ({BASE_MODEL_LICENSE}) |",
        f"| Files | {count} (see `{REPO_FILES_NAME}`), {total:,} bytes = {total / 1e9:.2f} GB |",
        "",
        f"`{REPO_FILES_NAME}` is the repository's file list at the pinned revision, fetched through the Hugging Face",
        f"API (`?blobs=true`, {repo_files['fetchedAt']}). `ForcedAlignerCatalogTests` pins that `ModelFileSelector` downloads exactly the nine",
        "files the loader reads (the safetensors, the index, `config.json`, `preprocessor_config.json`, `chat_template.json`,",
        "`generation_config.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`) and nothing else — not `README.md`, not",
        "`.gitattributes`. The fixture files keep unique basenames (`README-ForcedAligner.md`, `repo-files-ForcedAligner.json`,",
        "`golden-ForcedAligner.json`) because `MLXUITests` is an Xcode synchronized group that flattens resources by basename.",
        "",
    ]


def write_readme(out_dir: Path, golden: dict) -> None:
    lines = pinned_checkpoint_section(golden, load_repo_files())
    add = lines.append

    add(
        "Reference output of the **Python** `mlx-audio` forced aligner for the "
        "three clips below, regenerated by `make_align_golden.py`. The FA-4 "
        "parity test checks the Swift runner's units against `golden-ForcedAligner.json`. "
        "**Do not edit `golden-ForcedAligner.json` by hand** -- regenerate it."
    )
    add("")
    add("## Provenance")
    add("")
    add("| What | Value |")
    add("|---|---|")
    add(f"| HF model | `{golden['hfRepoId']}` |")
    add(f"| HF revision | `{golden['hfRevision']}` |")
    add(
        f"| Python reference | `mlx-audio` {golden['mlxAudioVersion']} (PyPI, "
        "installed by pip) |"
    )
    if golden["mlxAudioCommit"]:
        add(
            "| Reference commit | "
            f"[`{golden['mlxAudioCommit']}`]"
            f"(https://github.com/Blaizzy/mlx-audio/commit/{golden['mlxAudioCommit']}) "
            f"(upstream tag `{golden['mlxAudioTag']}`). The PyPI wheel ships no "
            "VCS metadata (`direct_url.json` absent, `INSTALLER` is `pip`), so "
            "the commit was established by sha256-matching every installed "
            "reference file against that tag; `golden-ForcedAligner.json` also pins those "
            "hashes. |"
        )
    else:
        add(
            "| Reference commit | not recorded (version not verified against "
            "an upstream tag); `golden-ForcedAligner.json` pins sha256 hashes of the "
            "reference sources. |"
        )
    add(
        f"| Reference sources sha256 | `mlx_audio/stt/models/qwen3_asr/"
        f"qwen3_forced_aligner.py` = `{golden['mlxAudioFileSha256']['stt/models/qwen3_asr/qwen3_forced_aligner.py']}` |"
    )
    add(
        f"| Checkpoint's own README | says the weights were converted with "
        f"mlx-audio **{golden['modelCardConversionVersion']}** from "
        f"`{BASE_MODEL}` |"
    )
    add(
        f"| Environment | Python {golden['pythonVersion']}, MLX "
        f"{golden['mlxVersion']}, transformers {golden['transformersVersion']}, "
        f"numpy {golden['numpyVersion']}, nagisa {golden['nagisaVersion']}, "
        f"{golden['platform']} |"
    )
    add("")
    add("## Clips")
    add("")
    add("| File | lang | voice (`say -r 180`) | text | units |")
    add("|---|---|---|---|---|")
    for clip in golden["clips"]:
        add(
            f"| `{clip['file']}` | `{clip['language']}` | {clip['voice']} | "
            f"{clip['text']} | {len(clip['items'])} |"
        )
    add("")
    add("The WAVs are committed because the folder stays under 2 MB; the goldens "
        "are therefore tied to the committed bytes (`fileSha256` in `golden-ForcedAligner.json`). "
        "The exact `say` commands are:")
    add("")
    add("```sh")
    for clip in golden["clips"]:
        spec = next(c for c in CLIPS if c["file"] == clip["file"])
        # Relative filename: the README is committed next to the WAVs.
        add(" ".join(say_command(spec, Path(spec["file"]))))
    add("```")
    add("")
    add("## Regenerate")
    add("")
    add("```sh")
    add("python3 -I make_align_golden.py --model-dir /private/tmp/fa-model")
    add("```")
    add("")
    add("Add `--force-audio` to overwrite the committed WAVs with fresh `say` "
        "output (this changes the goldens).")
    add("")
    add("## Units and gotchas")
    add("")
    add("- Times are **seconds**, in transcript order, rounded to three decimals "
        "by the reference itself. The reference does not clip times to the "
        "audio duration; the engine's contract (issue decision 4) does.")
    add("- One unit is one word for space-separated languages and one CJK "
        "character for Chinese, in both the Python and Swift splitter. "
        "Punctuation is dropped by the splitter, so `items` never contains it. "
        "Japanese differs -- see the next bullet.")
    add("- **Japanese is aligned with `language=\"Japanese\"`, the name the Swift "
        "engine passes, and the two runners split it differently.** Python takes "
        "its nagisa path and times `こんにちは、世界。` as "
        "`[\"こんにちは\", \"世界\"]`. The pinned Swift runner has no Japanese "
        "tokenizer (`Qwen3ForcedAligner.swift` @ `5c4c12b`, `encodeTimestamp`: "
        "every language but Chinese goes to `tokenizeSpaceLang`) and gives "
        "`[\"こんにちは\", \"世\", \"界\"]`. In the parity test only, for "
        "Japanese only, consecutive Swift units whose text exactly makes one "
        "reference unit are joined before the ±80 ms comparison; a join that does "
        "not line up fails. The engine and the app keep the library's split (issue "
        "#13 decision 4), and `ForcedAlignerEngineTests` pins it. The difference is "
        "reported upstream, with the Chinese drift (owner ruling 2), in "
        "[mlx-audio-swift issue 422](https://github.com/Neywa-Labs/mlx-audio-swift/issues/422).")
    add("  - Owner ruling 1 (Boo, 2026-11-27): \"Accept it only if it matches the "
        "language name the Swift engine passes. Regenerate the Japanese golden with "
        "the library's full name for Japanese. If the units and times are "
        "identical, keep it and note that in the fixture README. If they differ, "
        "the golden uses the same name Swift passes.\"")
    add("  - Owner ruling 8 (Boo, 2026-11-27): \"Option A. Use the 'Japanese' "
        "golden. In the parity test only, for Japanese only, join consecutive Swift "
        "units whose concatenated text exactly equals one reference unit, then "
        "apply ±80 ms to the joined span. If the join doesn't line up exactly, the "
        "test fails. The engine and the app's output stay as the library splits "
        "them (issue #13 decision 4: don't re-split). Add a test that pins Swift's "
        "current Japanese units (こんにちは, 世, 界), so a future upstream splitter "
        "shows up as a deliberate change. Note the difference in the fixture "
        "README, and add it to the same upstream mlx-audio-swift report as the "
        "Chinese drift (ruling 2). Chinese stays as ruling 2: exact units, "
        "±160 ms.\"")
    add("- Audio is read with the stdlib `wave` module and passed as a float32 "
        "array. `python3 -I` hides the user site-packages, so the package's "
        "miniaudio/cffi WAV loader raises `ModuleNotFoundError: _cffi_backend` "
        "if used; passing samples avoids it.")
    add("- The checkpoint's model card example (both CLI and Python) omits the "
        "transcript; for the aligner the transcript is mandatory, so that "
        "example fails. Always pass `--text` / `text=`.")
    add("")

    (out_dir / "README-ForcedAligner.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the Qwen3-ForcedAligner reference goldens (FA-4)."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/private/tmp/fa-model"),
        help="local checkpoint directory (never a repo id -- nothing is downloaded)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="where golden-ForcedAligner.json / README-ForcedAligner.md / WAVs live (default: script folder)",
    )
    parser.add_argument(
        "--force-audio",
        action="store_true",
        help="regenerate the committed WAVs with `say` (changes the goldens)",
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Python reference forced-alignment goldens", flush=True)
    print(f"  model:   {args.model_dir}", flush=True)
    print(f"  out:     {out_dir}", flush=True)
    print(f"  python:  {sys.version.split()[0]} (isolated: {sys.flags.isolated})", flush=True)

    print("clips:", flush=True)
    clip_paths = [generate_clip(clip, out_dir, args.force_audio) for clip in CLIPS]

    print("reference:", flush=True)
    model = load_reference(args.model_dir)

    print("align:", flush=True)
    started = time.time()
    clips = [align_clip(model, clip, path) for clip, path in zip(CLIPS, clip_paths)]
    total = time.time() - started

    golden = {
        "hfRepoId": HF_REPO_ID,
        "hfRevision": HF_REVISION,
        "units": (
            "reference splitter units in transcript order: whitespace word for "
            "space-separated languages, one CJK character for Chinese, nagisa "
            "words for Japanese (the Swift runner, mlx-audio-swift 5c4c12b, "
            "splits kana runs and single kanji; the parity test joins Swift units "
            "for Japanese only, owner ruling 8, 2026-11-27)"
        ),
        **reference_provenance(),
        "clips": clips,
    }

    golden_path = out_dir / "golden-ForcedAligner.json"
    golden_path.write_text(
        json.dumps(golden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_readme(out_dir, golden)

    print(f"wrote {golden_path}", flush=True)
    print(f"wrote {out_dir / 'README-ForcedAligner.md'}", flush=True)
    print(f"total alignment time: {total:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
