#!/usr/bin/env python3
"""MC5-1 spike: aggregate measurement JSONs and compare WAVs.

Outputs:
  out/measurements.json    canonical per-config/per-stage/duration table (Q8, Q6, Q12)
  out/audio_compare.json   objective WAV comparisons (bf16-vs-4bit, no-LM-vs-LM, repro)
  out/report_tables.md     the markdown tables used in the issue report
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

OUT = "/private/tmp/mc5/out"


def load_runs():
    runs = {}
    for f in sorted(glob.glob(os.path.join(OUT, "json", "*.json"))):
        if f.endswith(".log.json"):
            continue
        try:
            d = json.load(open(f))
        except Exception as exc:  # noqa: BLE001
            print("skip", f, exc)
            continue
        tag = os.path.basename(f)[:-5]
        runs[tag] = d
    return runs


def stage_max(d, names):
    out = {}
    for n in names:
        s = d.get("stages", {}).get(n)
        if s:
            out[n] = s
    return out


def canonical(runs):
    """Group runs into configs: (dit config, lm) -> duration -> row."""
    table = {}
    for tag, d in runs.items():
        c = d.get("config", {})
        key = f"{c.get('name')}|lm={c.get('lm')}"
        dur = int(c.get("duration_s", 0))
        row = {
            "tag": tag,
            "wall_gen_s": d.get("wall_s"),
            "init_wall_s": (d.get("init") or {}).get("wall_s"),
            "process_peak_mb": d.get("process_peak_footprint_mb"),
            "init_peak_mb": (d.get("init") or {}).get("peak_footprint_mb"),
            "time_costs": d.get("time_costs", {}),
            "stage_peaks_mb": {
                n: (s or {}).get("peak_footprint_mb") for n, s in stage_max(
                    d,
                    [
                        "text.encode",
                        "condition.prepare",
                        "dit.diffusion",
                        "vae.decode.mlx",
                        "lm.load",
                        "lm.generate",
                    ],
                ).items()
            },
            "stage_wall_s": {
                n: (s or {}).get("total_s") for n, s in stage_max(
                    d,
                    ["text.encode", "condition.prepare", "dit.diffusion", "vae.decode.mlx", "lm.load", "lm.generate"],
                ).items()
            },
            "torch_mps_driver_mb": round((d.get("torch_mem_end") or {}).get("torch_mps_driver", 0) / 2**20, 1),
            "torch_mps_allocated_mb": round((d.get("torch_mem_end") or {}).get("torch_mps_allocated", 0) / 2**20, 1),
            "mlx_peak_mb": round((d.get("mlx_mem_end") or {}).get("mlx_peak", 0) / 2**20, 1),
            "lm_metadata": d.get("lm_metadata"),
            "wavs": [w.get("saved") for w in d.get("wavs", []) if w.get("saved")],
        }
        table.setdefault(key, {})[dur] = row
    return table


# ------------------------------------------------------------------ audio compare
def load_wav(path):
    import soundfile as sf

    x, sr = sf.read(path, always_2d=True)
    return np.asarray(x, dtype=np.float32), sr


def log_mel(x, sr, n_mels=80, n_fft=2048, hop=512):
    import torch
    import torchaudio

    mono = torch.from_numpy(x.mean(axis=1))
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0
    )(mono)
    return torch.log(mel.clamp_min(1e-10))


def compare(a_path, b_path):
    a, sr_a = load_wav(a_path)
    b, sr_b = load_wav(b_path)
    res = {
        "a": a_path,
        "b": b_path,
        "sr_a": sr_a,
        "sr_b": sr_b,
        "dur_a": round(a.shape[0] / sr_a, 3),
        "dur_b": round(b.shape[0] / sr_b, 3),
        "channels_a": a.shape[1],
        "channels_b": b.shape[1],
        "rms_a": float(np.sqrt(np.mean(a**2))),
        "rms_b": float(np.sqrt(np.mean(b**2))),
    }
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return res
    a = a[:n]
    b = b[:n]
    ma = log_mel(a, sr_a)
    mb = log_mel(b, sr_b)
    m = min(ma.shape[1], mb.shape[1])
    ma, mb = ma[:, :m], mb[:, :m]
    va = ma.flatten().detach().cpu().numpy()
    vb = mb.flatten().detach().cpu().numpy()
    res["logmel_cosine"] = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
    res["logmel_lsd"] = float(np.sqrt(np.mean((va - vb) ** 2)))
    # waveform correlation on the quieter of the two channels is not meaningful for
    # different weights; use a short-time energy-envelope correlation instead.
    e_a = np.sqrt(np.convolve(a.mean(axis=1) ** 2, np.ones(1024) / 1024, mode="valid")[::512])
    e_b = np.sqrt(np.convolve(b.mean(axis=1) ** 2, np.ones(1024) / 1024, mode="valid")[::512])
    k = min(len(e_a), len(e_b))
    res["envelope_corr"] = float(
        np.corrcoef(e_a[:k], e_b[:k])[0, 1] if k > 2 and e_a.std() > 0 and e_b.std() > 0 else 0.0
    )
    # Peak / crest factor
    res["peak_a"] = float(np.max(np.abs(a)))
    res["peak_b"] = float(np.max(np.abs(b)))
    return res


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    runs = load_runs()
    table = canonical(runs)
    payload = {
        "host": {
            "chip": "Apple M3 Pro",
            "ram_gib": 18.0,
            "macos": next(iter(runs.values()))["host"].get("macos") if runs else None,
            "torch": next(iter(runs.values()))["host"].get("torch") if runs else None,
            "mlx_python": next(iter(runs.values()))["host"].get("mlx_python") if runs else None,
            "transformers": next(iter(runs.values()))["host"].get("transformers") if runs else None,
            "mlx_lm": next(iter(runs.values()))["host"].get("mlx_lm") if runs else None,
            "python": next(iter(runs.values()))["host"].get("python") if runs else None,
        },
        "notes": [
            "process_peak_mb is the whole-process phys_footprint peak sampled at 50 ms.",
            "torch_mps_driver_mb is torch's MPS allocator footprint at process end; it stays "
            "resident after tensors are freed (measured: allocating+freeing 2 GiB still leaves "
            "~2.1 GiB of footprint), so it is measurement overhead a torch-free Swift port will not have.",
            "stage_peaks_mb is the peak phys_footprint inside that stage window, so it includes "
            "every other resident component at that time.",
        ],
        "configs": table,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "measurements.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("wrote measurements.json; configs:", list(table.keys()))

    # ---------------------------------------------------------------- WAV compares
    cmp = {}
    pairs = [
        ("q8-bf16-vs-4bit-30s", "q8-bf16-nolm-30s", "q8-4bit-nolm-30s"),
        ("q4-instr-nolm-vs-lm06", "q4-nolm-instrumental", "q4-lm06-instrumental"),
        ("q4-instr-nolm-vs-lm17", "q4-nolm-instrumental", "q4-lm17-instrumental"),
        ("q4-instr-lm06-vs-lm17", "q4-lm06-instrumental", "q4-lm17-instrumental"),
        ("q4-english-nolm-vs-lm06", "q4-nolm-english_vocal", "q4-lm06-english_vocal"),
        ("q4-chinese-nolm-vs-lm06", "q4-nolm-chinese_vocal", "q4-lm06-chinese_vocal"),
        ("q4-lm06-instrumental-repro", "q4-lm06-instrumental", "q4-lm06-instrumental-repro"),
    ]
    for name, a_tag, b_tag in pairs:
        a = runs.get(a_tag, {}).get("wavs", [{}])[0].get("saved")
        b = runs.get(b_tag, {}).get("wavs", [{}])[0].get("saved")
        if not a or not b or not (os.path.exists(a) and os.path.exists(b)):
            cmp[name] = {"missing": True, "a": a, "b": b}
            continue
        try:
            c = compare(a, b)
            c["sha_a"] = sha(a)
            c["sha_b"] = sha(b)
            c["identical_bytes"] = c["sha_a"] == c["sha_b"]
            cmp[name] = c
            print(name, {k: c.get(k) for k in ("logmel_cosine", "logmel_lsd", "envelope_corr", "identical_bytes")})
        except Exception as exc:  # noqa: BLE001
            cmp[name] = {"error": repr(exc)}
            print("compare failed", name, exc)
    with open(os.path.join(OUT, "audio_compare.json"), "w") as f:
        json.dump(cmp, f, indent=2)
    print("wrote audio_compare.json")

    # ------------------------------------------------------------ Q8 markdown
    lines = []
    for key, durs in table.items():
        cfg, lm = key.split("|")
        if lm == "lm=1.7B":
            # Q8's third configuration is the 4-bit DiT; the 1.7B LM row (30 s, batch-2) is Q4 audio
            # evidence and stays in measurements.json, not in the Q8 memory/time table.
            continue
        lines.append(f"### {cfg} / {lm}\n")
        lines.append(
            "| dur | init wall | text enc s/peak | condition s/peak | DiT 8 steps s/peak | VAE decode s/peak | gen wall (model) | process peak | MLX peak | torch@exit |"
        )
        lines.append("|---:|---:|---|---|---|---|---:|---:|---:|---:|")
        for dur in sorted(durs):
            r = durs[dur]
            sp = r["stage_peaks_mb"]
            sw = r["stage_wall_s"]

            def cell(name):
                if name not in sp or sp[name] is None:
                    return "—"
                return f"{sw.get(name, 0):.2f} s / {sp[name]/1024:.2f} GB"

            lines.append(
                "| {} s | {} s | {} | {} | {} | {} | {} s | {:.2f} GB | {:.1f} GB | {:.1f} GB |".format(
                    dur,
                    f"{r['init_wall_s']:.1f}" if r.get("init_wall_s") else "—",
                    cell("text.encode"),
                    cell("condition.prepare"),
                    cell("dit.diffusion"),
                    cell("vae.decode.mlx"),
                    f"{r['wall_gen_s']:.1f}" if r.get("wall_gen_s") else "—",
                    (r["process_peak_mb"] or 0) / 1024,
                    (r["mlx_peak_mb"] or 0) / 1024,
                    (r["torch_mps_driver_mb"] or 0) / 1024,
                )
            )
        lines.append("")
    with open(os.path.join(OUT, "q8_tables.md"), "w") as f:
        f.write("\n".join(lines))
    print("wrote q8_tables.md")


if __name__ == "__main__":
    sys.exit(main())
