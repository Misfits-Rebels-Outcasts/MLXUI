# MC5-1 spike — ACE-Step 1.5 → MLX Swift port plan

Artifacts for issue **#10** ("Spike: plan an MLX Swift port of ACE-Step 1.5 for commercial-use
music generation (MC5)"). The report is the issue comments; this directory is the reproducible
evidence behind it.

Pinned inputs:

| Item | Revision |
|---|---|
| Weights `ACE-Step/Ace-Step1.5` | `19671f406d603126926c1b7e2adc169acbcade22` |
| Upstream code `ace-step/ACE-Step-1.5` | `ca1e85fe9430179831e6bc6be790c332190a3866` |
| `mlx-swift` / `mlx-swift-lm` | `0.32.3` (`1960120`) / `3.32.3` (`3b339ad`) |

Machine: **Apple M3 Pro, 12 cores, 18 GiB RAM**, macOS 27.0, Xcode 27.2 / Swift 6.4.2.
Upstream Python env: `uv sync --frozen` → Python 3.12.10, `mlx` 0.30.6, `mlx-lm` 0.29.1,
`torch` 2.10.0, `transformers` 4.57.6.

## Contents

| Path | What |
|---|---|
| `findings/` | the research notes the report was assembled from: component map, weights inventory, text encoder, licence, MLX-Swift gap table, parity plan, slice plan, headless run notes |
| `findings/inventory/` | measured safetensors inventory (names, shapes, dtypes, per-module totals) produced by `scripts/weights/inventory_safetensors.py` |
| `scripts/weights/` | stdlib-only safetensors parser + group totals (`python3 -I`, no torch/numpy) |
| `scripts/mc5_measure.py` | measurement harness: runs the pinned upstream pipeline on macOS/MLX, records per-stage wall time and peak `phys_footprint`, saves WAVs and JSON |
| `scripts/mc5_vae_probe.py` | isolated MLX-VAE decode probe (time/memory per duration, f16 vs f32) |
| `scripts/mc5_analyze.py` | aggregates the JSON results into `results/measurements.json` and computes objective WAV comparisons |
| `scripts/run_matrix.sh` | the exact matrix (Q8 × 3 configs × 3 durations; Q4 × 3 prompts × 3 LM settings) |
| `swift-probe/` | SPM executable built against the pinned `mlx-swift` checkout; 21 op-level parity checks vs Python MLX (`python_checks.py`, `compare.py`, `swift-out-final.txt`) |
| `results/` | `measurements.json` (canonical table), `audio_compare.json`, VAE probe results, matrix log, per-run JSONs |
| `samples/mc5-samples.zip` | the representative WAVs: Q4's 3 prompts × {no-LM, 0.6B, 1.7B} at 30 s + the bf16-vs-4-bit pair at 30 s, same seed |

No weight files are committed or attached anywhere.

## Reproducing the measurements

```bash
# 1. pinned upstream checkout + weights (outside the app repo)
git clone https://github.com/ace-step/ACE-Step-1.5.git /private/tmp/mc5/ACE-Step-1.5
git -C /private/tmp/mc5/ACE-Step-1.5 checkout ca1e85fe9430179831e6bc6be790c332190a3866
hf download ACE-Step/Ace-Step1.5 \
  --revision 19671f406d603126926c1b7e2adc169acbcade22 --local-dir /private/tmp/mc5/Ace-Step1.5
# (Q4's 0.6B planner is a separate repo, ~1.37 GB)
hf download ACE-Step/acestep-5Hz-lm-0.6B \
  --local-dir /private/tmp/mc5/Ace-Step1.5/acestep-5Hz-lm-0.6B

# 2. upstream environment (pinned by uv.lock)
cd /private/tmp/mc5/ACE-Step-1.5 && uv sync --frozen

# 3. run the matrix (sequential; memory-bound)
bash /path/to/MC5-spike/scripts/run_matrix.sh

# 4. aggregate + compare
/private/tmp/mc5/ACE-Step-1.5/.venv/bin/python scripts/mc5_analyze.py
```

The weights inventory scripts are stdlib-only and can be run before any Python environment exists:

```bash
python3 -I scripts/weights/inventory_safetensors.py /private/tmp/mc5/Ace-Step1.5
```

## Measurement-method notes (the parts that matter for reading the numbers)

- `phys_footprint` is read from `proc_pid_rusage(RUSAGE_INFO_V4)` for the measured process only,
  sampled every 50 ms; a "stage peak" is the maximum inside that stage's window, so it includes
  every component resident at that moment.
- Upstream's macOS path is a **torch + MLX hybrid**, and on MPS it (a) loads the DiT in float32,
  (b) builds a second float32 MLX copy and (c) never releases the torch decoder — ~19 GB of DiT,
  which does not fit 18 GiB. Two **measurement-only** patches were therefore applied for the
  `bf16` / `bf16-4bit` configurations (the exact code is in `scripts/mc5_measure.py`):
  1. load the torch condition-encoder half in bf16 and release the torch DiT decoder immediately;
  2. build the native MLX DiT decoder **straight from the official safetensors** in bf16 (this is
     the loader the Swift port has to write, and it removes the need for torch at DiT load time).
  `bf16-4bit` additionally quantizes that MLX decoder with `mlx.nn.quantize(group_size=64, bits=4)`.
  The `upstream` configuration (no patches) was attempted and is reported only as "does not fit".
- **torch's MPS allocator keeps freed memory resident.** A controlled check on this Mac:
  allocating then freeing a 2 GiB MPS tensor and calling `torch.mps.empty_cache()` leaves the
  process footprint ~2.1 GiB higher — the pages are not returned to the OS. That retained memory
  is inside every whole-process number here and is **not** part of a torch-free Swift port; the
  report states the port projection separately from the measured number.
- `ACESTEP_MLX_VAE_FP16=1` is set for the matrix: f32 MLX VAE decode did not complete 4-minute
  audio on this machine in a reasonable time (process reached 13.84 GB with 5.3 GB swap). The
  f16 path is an upstream-supported switch; f32/f16 decode numbers are in `results/`.
- No-LM runs set `thinking=False` and all four `use_cot_*` flags False; LM runs use
  `batch_size=2, seeds=[s,s+1]`. Upstream's **single-item LM decoders have no seeding at all**
  (`llm_inference.py:985/3397/3720`); the batch path seeds the **codes phase** per item, but the
  **CoT metadata phase is generated once, unseeded, even in batch mode** — so a fixed seed pins
  the DiT and the code stream, not the LM's caption/metadata. The duplicate `q4-lm06-instrumental-repro`
  run is included to show what that means in practice.
