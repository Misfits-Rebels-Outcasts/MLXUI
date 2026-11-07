#!/usr/bin/env python3
"""MC5-1 spike: MLX VAE decode probe.

Measures per-duration MLX VAE decode wall time and peak footprint, and how the
chunk size / fp16 switch affects both. Q6.

Run with the upstream uv env:
  .venv/bin/python /private/tmp/mc5/scripts/mc5_vae_probe.py --out out/vae_probe.json

Environment knobs honoured: ACESTEP_MLX_VAE_CHUNK (default from gpu_config), ACESTEP_MLX_VAE_FP16.
"""
import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, "/private/tmp/mc5/scripts")
REPO = "/private/tmp/mc5/ACE-Step-1.5"
CKPT = "/private/tmp/mc5/Ace-Step1.5"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/private/tmp/mc5/out/vae_probe.json")
    ap.add_argument("--lens", default="250,750,1500,3000,6000")
    ap.add_argument("--free-torch", action="store_true", default=True)
    args = ap.parse_args()

    os.environ.setdefault("ACESTEP_CHECKPOINTS_DIR", CKPT)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, REPO)

    import mc5_measure as M

    sampler = M.Sampler(interval=0.02)
    sampler.start()
    import torch

    M.apply_patches("bf16", sampler)
    from acestep.gpu_config import get_gpu_config, set_global_gpu_config

    set_global_gpu_config(get_gpu_config())
    from acestep.handler import AceStepHandler

    t0 = time.monotonic()
    dit = AceStepHandler()
    status, ok = dit.initialize_service(
        project_root=REPO,
        config_path="acestep-v15-turbo",
        device="mps",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
        quantization=None,
        use_mlx_dit=True,
    )
    init_s = time.monotonic() - t0
    if not ok:
        print(status)
        return 1
    init_fp = M.phys_footprint()

    if args.free_torch:
        dit.text_encoder = None
        dit.text_tokenizer = None
        dit.model = None
        dit.vae = None
        gc.collect()
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    base_fp = M.phys_footprint()

    results = {}
    for s in args.lens.split(","):
        T = int(s)
        seconds = T / 25.0
        lat = torch.randn(1, 64, T) * 0.1
        try:
            import mlx.core as mx

            mx.reset_peak_memory()
            mx.clear_cache()
        except Exception:
            pass
        fp0 = M.phys_footprint()
        t1 = time.monotonic()
        out = dit._mlx_vae_decode(lat)
        if hasattr(out, "detach"):
            out = out.detach()
        try:
            import mlx.core as mx

            mx.eval(out) if hasattr(mx, "eval") and not hasattr(out, "detach") else None
        except Exception:
            pass
        wall = time.monotonic() - t1
        fp1 = M.phys_footprint()
        mem = M._mlx_mem()
        shape = tuple(out.shape) if hasattr(out, "shape") else None
        results[str(T)] = {
            "latent_frames": T,
            "seconds": round(seconds, 1),
            "wall_s": round(wall, 2),
            "peak_footprint_mb": round(max(fp0, fp1) / 2**20, 1),
            "footprint_delta_mb": round((fp1 - fp0) / 2**20, 1),
            "mlx_peak_mb": round(mem.get("mlx_peak", 0) / 2**20, 1),
            "out_shape": shape,
        }
        print(results[str(T)])

    payload = {
        "host": {
            "chip": os.popen("sysctl -n machdep.cpu.brand_string").read().strip(),
            "ram_gib": round(int(os.popen("sysctl -n hw.memsize").read().strip()) / 2**30, 2),
        },
        "env": {
            "ACESTEP_MLX_VAE_CHUNK": os.environ.get("ACESTEP_MLX_VAE_CHUNK"),
            "ACESTEP_MLX_VAE_FP16": os.environ.get("ACESTEP_MLX_VAE_FP16"),
        },
        "init_wall_s": round(init_s, 2),
        "init_footprint_mb": round(init_fp / 2**20, 1),
        "base_footprint_mb_after_free": round(base_fp / 2**20, 1),
        "mlx_vae_chunk_size": getattr(dit, "mlx_vae_chunk_size", None),
        "vae_dtype": str(getattr(dit, "_mlx_vae_dtype", "?")),
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
