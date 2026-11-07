#!/usr/bin/env python3
"""MC5-1 spike: single-run measurement harness for ACE-Step 1.5 on macOS/MLX.

Runs the *pinned upstream* pipeline (ACE-Step-1.5 @ ca1e85f) on this Mac and records
per-stage wall time and peak phys_footprint, then writes one JSON result.

Configurations (--config):
  upstream   : upstream defaults exactly as start_gradio_ui_macos.sh sets them on this
               Mac (torch DiT loads in float32, MLX DiT copy is float32, both stay
               resident). Expected to exceed 18 GiB physical memory -> swap.
  bf16       : upstream path with the DiT loaded in bfloat16 both in PyTorch and in the
               native MLX decoder, and the redundant PyTorch decoder released after the
               MLX copy is built (the "staged, one copy" variant decision 4 describes).
  bf16-4bit  : bf16 + quantize the native MLX decoder to 4-bit (group 64) after load,
               from the official weights.

Everything is measured in-process on this machine. The script never downloads anything.

Usage:
  .venv/bin/python /private/tmp/mc5/scripts/mc5_measure.py \
      --config bf16 --lm none --duration 30 --seed 12345 \
      --prompt-file /private/tmp/mc5/scripts/prompts.json --prompt-id instrumental \
      --out /private/tmp/mc5/out/measure.json --wav-dir /private/tmp/mc5/out/wav
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import ctypes
import gc
import json
import os
import platform
import struct
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

REPO = "/private/tmp/mc5/ACE-Step-1.5"
CKPT = "/private/tmp/mc5/Ace-Step1.5"

# ---------------------------------------------------------------- footprint probe
_lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
_lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_lib.proc_pid_rusage.restype = ctypes.c_int
_RUSAGE_INFO_V4 = 4
_rusage_buf = ctypes.create_string_buffer(2048)


def phys_footprint() -> int:
    """Current process physical footprint in bytes (Activity Monitor's 'Memory')."""
    rc = _lib.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(_rusage_buf))
    if rc != 0:
        return -1
    # rusage_info_v2/v4 prefix: 16-byte uuid + 8 uint64 fields; phys_footprint is the
    # 8th uint64 after the uuid (ri_user_time, ri_system_time, ri_pkg_idle_wkups,
    # ri_interrupt_wkups, ri_pageins, ri_wired_size, ri_resident_size, ri_phys_footprint).
    return struct.unpack_from("<Q", _rusage_buf.raw, 16 + 7 * 8)[0]


def _mlx_mem() -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        import mlx.core as mx

        for name, fn in (
            ("mlx_active", getattr(mx, "get_active_memory", None)),
            ("mlx_peak", getattr(mx, "get_peak_memory", None)),
            ("mlx_cache", getattr(mx, "get_cache_memory", None)),
        ):
            if fn is not None:
                try:
                    out[name] = int(fn())
                except Exception:
                    pass
    except Exception:
        pass
    return out


def _torch_mem() -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        import torch

        if torch.backends.mps.is_available():
            out["torch_mps_allocated"] = int(torch.mps.current_allocated_memory())
            out["torch_mps_driver"] = int(torch.mps.driver_allocated_memory())
    except Exception:
        pass
    return out


class Sampler:
    """Background thread sampling phys_footprint; stage windows tag the samples."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.samples: List[Tuple[float, int]] = []
        self.events: List[Tuple[str, str, float, float]] = []  # stage, kind, start, end
        self.marks: List[Tuple[float, str]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()
        self.t0 = time.monotonic()
        self.max_fp = 0

    def _run(self) -> None:
        while not self._stop.is_set():
            fp = phys_footprint()
            t = time.monotonic()
            with self._lock:
                self.samples.append((t, fp))
                if fp > self.max_fp:
                    self.max_fp = fp
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    @contextlib.contextmanager
    def stage(self, name: str):
        start = time.monotonic()
        fp0 = phys_footprint()
        try:
            yield
        finally:
            end = time.monotonic()
            fp1 = phys_footprint()
            with self._lock:
                self.events.append((name, "window", start, end))
                self.events.append((f"{name}.snapshot", "point", fp1, fp1))
            self._mark(f"{name}: {end - start:.2f}s fp={fp1 / 2**20:.1f}MB (delta {(fp1 - fp0) / 2**20:+.1f}MB)")

    def _mark(self, text: str) -> None:
        self.marks.append((time.monotonic() - self.t0, text))

    def stage_peak(self, name: str) -> Dict[str, Any]:
        with self._lock:
            windows = [(s, e) for (n, k, s, e) in self.events if n == name and k == "window"]
            samples = list(self.samples)
        if not windows:
            return {}
        total = sum(e - s for s, e in windows)
        peak = 0
        peak_t = 0.0
        for s, e in windows:
            for t, fp in samples:
                if s - 0.02 <= t <= e + 0.02 and fp > peak:
                    peak, peak_t = fp, t
        fp_start = next((fp for t, fp in samples if t >= windows[0][0]), 0)
        return {
            "count": len(windows),
            "total_s": round(total, 3),
            "peak_footprint_mb": round(peak / 2**20, 1),
            "start_footprint_mb": round(fp_start / 2**20, 1),
            "peak_delta_mb": round((peak - fp_start) / 2**20, 1),
        }


# ------------------------------------------------------------------- log capture
LOGS: List[Dict[str, Any]] = []


def install_log_sink() -> None:
    try:
        from loguru import logger

        def sink(message):
            rec = message.record
            LOGS.append(
                {
                    "t": round(rec["time"].timestamp() - _T0, 3),
                    "level": rec["level"].name,
                    "msg": rec["message"],
                }
            )

        logger.remove()
        logger.add(sink, level=os.environ.get("MC5_LOG_LEVEL", "INFO"))
        logger.add(sys.stderr, level=os.environ.get("MC5_LOG_LEVEL", "INFO"))
    except Exception:
        pass


_T0 = time.time()
STAGED_FREE = False

# ---------------------------------------------------------------------- patching


def _module_bytes_torch(module) -> Dict[str, int]:
    out = {"params_bytes": 0, "params": 0}
    try:
        for p in module.parameters():
            out["params_bytes"] += p.numel() * p.element_size()
            out["params"] += p.numel()
    except Exception:
        pass
    return out


def _module_bytes_mlx(module) -> Dict[str, int]:
    out = {"params_bytes": 0, "params": 0}
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        for _, v in tree_flatten(module.parameters()):
            out["params_bytes"] += v.nbytes
            out["params"] += v.size
    except Exception:
        pass
    return out


def apply_patches(config: str, sampler: Sampler) -> List[str]:
    """Apply measurement-time patches; returns human-readable list of what was done."""
    notes: List[str] = []
    import torch

    from acestep.core.generation.handler.init_service_loader import InitServiceLoaderMixin
    from acestep.core.generation.handler.mlx_dit_init import MlxDitInitMixin

    if config in ("bf16", "bf16-4bit"):
        st_path = os.path.join(CKPT, "acestep-v15-turbo", "model.safetensors")

        # 1. Load the PyTorch side (condition encoder + attention-pooler/tokenizer) in
        #    bfloat16. Upstream on MPS hard-codes float32 (init_service_orchestrator.py:103)
        #    which doubles the 4.8 GB checkpoint. The DiT *decoder* is released immediately
        #    after load: the native MLX decoder is built straight from the official
        #    safetensors, so the PyTorch decoder is never needed.
        orig_load = InitServiceLoaderMixin._load_main_model_from_checkpoint

        def load_bf16(self, **kwargs):
            self.dtype = torch.bfloat16
            res = orig_load(self, **kwargs)
            try:
                delattr(self.model, "decoder")
                notes.append("loaded torch condition encoder in bfloat16; released redundant torch DiT decoder")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"could not release torch decoder: {exc!r}")
            gc.collect()
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
            return res

        InitServiceLoaderMixin._load_main_model_from_checkpoint = load_bf16

        # 2. Build the native MLX DiT decoder directly from the official safetensors in
        #    bfloat16, applying the same three layout conversions as
        #    acestep/models/mlx/dit_convert.py:44-66. This is exactly the loader a Swift
        #    port has to write (decision 2: no torch at load time).
        def mlx_dit_from_safetensors(self, compile_model: bool = False) -> bool:
            from acestep.models.mlx import mlx_available

            if not mlx_available():
                return False
            import mlx.core as mx
            from acestep.models.mlx.dit_model import MLXDiTDecoder

            dec = MLXDiTDecoder.from_config(self.config)
            raw = mx.load(st_path)
            weights = []
            for k, v in raw.items():
                if not k.startswith("decoder."):
                    continue
                k2 = k[len("decoder.") :]
                if "rotary_emb" in k2:
                    continue
                if k2.startswith("proj_in.1."):
                    k2 = k2.replace("proj_in.1.", "proj_in.")
                    if k2.endswith(".weight"):
                        v = v.swapaxes(1, 2)
                elif k2.startswith("proj_out.1."):
                    k2 = k2.replace("proj_out.1.", "proj_out.")
                    if k2.endswith(".weight"):
                        v = v.transpose(1, 2, 0)
                weights.append((k2, v))
            dec.load_weights(weights)
            mx.eval(dec.parameters())
            notes.append(f"native MLX DiT decoder built from {st_path} ({len(weights)} tensors, bfloat16)")
            if config == "bf16-4bit":
                import mlx.nn as nn

                nn.quantize(dec, group_size=64, bits=4)
                mx.eval(dec.parameters())
                notes.append("MLX DiT decoder quantized to 4-bit (group_size=64) from official weights")
            self.mlx_decoder = dec
            self.use_mlx_dit = True
            self.mlx_dit_compiled = False
            return True

        MlxDitInitMixin._init_mlx_dit = mlx_dit_from_safetensors

    return notes


# --------------------------------------------------------------- stage wrapping


def wrap_class_stages(sampler: Sampler) -> None:
    """Wrap methods that exist before the model instance is created."""
    from acestep.core.generation.handler.conditioning_embed import ConditioningEmbedMixin
    from acestep.core.generation.handler.diffusion import DiffusionMixin
    from acestep.core.generation.handler.generate_music import GenerateMusicMixin
    from acestep.core.generation.handler.init_service_loader import InitServiceLoaderMixin
    from acestep.core.generation.handler.init_service_loader_components import (
        InitServiceLoaderComponentsMixin,
    )
    from acestep.core.generation.handler.init_service_orchestrator import (
        InitServiceOrchestratorMixin,
    )
    from acestep.core.generation.handler.mlx_dit_init import MlxDitInitMixin
    from acestep.core.generation.handler.mlx_vae_decode_native import MlxVaeDecodeNativeMixin
    from acestep.core.generation.handler.mlx_vae_init import MlxVaeInitMixin
    from acestep.core.generation.handler.vae_decode import VaeDecodeMixin
    from acestep.llm_inference import LLMHandler

    def wrap(cls, name: str, label: str) -> None:
        orig = getattr(cls, name)

        @functools.wraps(orig)
        def wrapped(*args, **kwargs):
            with sampler.stage(label):
                return orig(*args, **kwargs)

        setattr(cls, name, wrapped)

    wrap(InitServiceOrchestratorMixin, "initialize_service", "init")
    wrap(InitServiceLoaderMixin, "_load_main_model_from_checkpoint", "init.load_torch_dit")
    wrap(InitServiceLoaderComponentsMixin, "_load_vae_model", "init.load_torch_vae")
    wrap(InitServiceLoaderComponentsMixin, "_load_text_encoder_and_tokenizer", "init.load_text_encoder")
    wrap(MlxDitInitMixin, "_init_mlx_dit", "init.mlx_dit")
    wrap(MlxVaeInitMixin, "_init_mlx_vae", "init.mlx_vae")
    wrap(LLMHandler, "initialize", "lm.load")
    wrap(LLMHandler, "generate_with_stop_condition", "lm.generate")
    wrap(ConditioningEmbedMixin, "infer_text_embeddings", "text.encode")
    wrap(ConditioningEmbedMixin, "infer_lyric_embeddings", "lyric.encode")
    wrap(GenerateMusicMixin, "generate_music", "generate.total")

    orig_diff = DiffusionMixin._mlx_run_diffusion

    @functools.wraps(orig_diff)
    def diffusion_staged(*args, **kwargs):
        with sampler.stage("dit.diffusion"):
            out = orig_diff(*args, **kwargs)
        self = args[0]
        if STAGED_FREE and not getattr(self, "_mc5_staged_freed", False):
            try:
                import torch

                self.text_encoder = None
                self.text_tokenizer = None
                self.model = None
                self._mc5_staged_freed = True
                gc.collect()
                torch.mps.empty_cache()
                sampler._mark("staged free: torch text encoder + condition encoder released before VAE decode")
            except Exception as exc:  # noqa: BLE001
                sampler._mark(f"staged free failed: {exc!r}")
        return out

    DiffusionMixin._mlx_run_diffusion = diffusion_staged
    wrap(VaeDecodeMixin, "tiled_decode", "vae.decode.tiled")
    wrap(MlxVaeDecodeNativeMixin, "_mlx_vae_decode", "vae.decode.mlx")


def wrap_instance_stages(sampler: Sampler, model, staged_free: bool = False) -> None:
    """Wrap the remote-code model's prepare_condition on the instance."""
    original_prepare = model.prepare_condition

    @functools.wraps(original_prepare)
    def prepare_wrapped(*args, **kwargs):
        with sampler.stage("condition.prepare"):
            return original_prepare(*args, **kwargs)

    model.prepare_condition = prepare_wrapped


# ------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["upstream", "bf16", "bf16-4bit"], required=True)
    ap.add_argument("--lm", choices=["none", "0.6B", "1.7B"], default="none")
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--prompt-file", default="/private/tmp/mc5/scripts/prompts.json")
    ap.add_argument("--prompt-id", default="instrumental")
    ap.add_argument("--caption", default=None, help="override prompt caption")
    ap.add_argument("--lyrics", default=None, help="override lyrics")
    ap.add_argument("--instrumental", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument(
        "--staged-free",
        action="store_true",
        help="emulate decision-4 staging: free the PyTorch components once their output is computed",
    )
    ap.add_argument("--thinking", choices=["auto", "on", "off"], default="auto")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    os.makedirs(args.wav_dir, exist_ok=True)

    # ---- environment: everything local, pinned
    os.environ.setdefault("ACESTEP_CHECKPOINTS_DIR", CKPT)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    sys.path.insert(0, REPO)

    prompts = {}
    if os.path.exists(args.prompt_file):
        with open(args.prompt_file) as f:
            prompts = json.load(f)
    prompt = prompts.get(args.prompt_id, {})
    caption = args.caption if args.caption is not None else prompt.get("caption", "")
    lyrics = args.lyrics if args.lyrics is not None else prompt.get("lyrics", "")
    instrumental = bool(args.instrumental or prompt.get("instrumental", False))
    vocal_language = prompt.get("vocal_language", "unknown")

    THINKING = {"auto": None, "on": True, "off": False}[args.thinking]

    sampler = Sampler()
    install_log_sink()
    import torch
    import mlx.core as mx
    import transformers

    sampler.start()

    result: Dict[str, Any] = {
        "ticket": "MC5-1 spike / MLXUI#10",
        "pinned": {
            "hf_repo": "ACE-Step/Ace-Step1.5 @ 19671f406d603126926c1b7e2adc169acbcade22",
            "upstream_repo": "ace-step/ACE-Step-1.5 @ ca1e85fe9430179831e6bc6be790c332190a3866",
            "mlx_swift": "0.32.3 (1960120)",
            "mlx_swift_lm": "3.32.3 (3b339ad)",
        },
        "host": {},
        "config": {
            "name": args.config,
            "lm": args.lm,
            "duration_s": args.duration,
            "seed": args.seed,
            "steps": args.steps,
            "shift": args.shift,
            "caption": caption,
            "lyrics": lyrics,
            "instrumental": instrumental,
            "vocal_language": vocal_language,
            "staged_free": bool(args.staged_free),
            "batch": int(args.batch),
        },
        "stages": {},
        "time_costs": {},
        "patches": [],
        "wavs": [],
        "ok": False,
    }

    try:
        try:
            import psutil  # optional

            vm = psutil.virtual_memory()
            result["host"] = {
                "chip": os.popen("sysctl -n machdep.cpu.brand_string").read().strip(),
                "model": os.popen("sysctl -n hw.model").read().strip(),
                "ram_bytes": int(os.popen("sysctl -n hw.memsize").read().strip()),
                "ram_gib": round(int(os.popen("sysctl -n hw.memsize").read().strip()) / 2**30, 2),
                "macos": platform.mac_ver()[0],
                "kernel": platform.release(),
            }
            del vm
        except Exception:
            pass
        result["host"].update(
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "mlx_python": getattr(mx, "__version__", "unknown"),
                "transformers": transformers.__version__,
                "milestone": "MC5-1",
            }
        )
        try:
            import mlx_lm

            result["host"]["mlx_lm"] = getattr(mlx_lm, "__version__", "unknown")
        except Exception:
            pass
        try:
            import safetensors

            result["host"]["safetensors"] = safetensors.__version__
        except Exception:
            pass
        try:
            result["host"]["uv_lock"] = os.popen(f"grep -m1 'version = ' {REPO}/uv.lock").read().strip()
        except Exception:
            pass

        # ---- apply measurement patches before importing/initialising handlers
        result["patches"] = apply_patches(args.config, sampler)

        from acestep.gpu_config import get_gpu_config, set_global_gpu_config

        gcfg = get_gpu_config()
        set_global_gpu_config(gcfg)
        result["gpu_config"] = {
            "tier": gcfg.tier,
            "gpu_memory_gb": round(gcfg.gpu_memory_gb, 2),
            "max_duration_with_lm": gcfg.max_duration_with_lm,
            "max_duration_without_lm": gcfg.max_duration_without_lm,
            "recommended_backend": gcfg.recommended_backend,
            "offload_to_cpu_default": gcfg.offload_to_cpu_default,
            "offload_dit_to_cpu_default": gcfg.offload_dit_to_cpu_default,
        }

        from acestep.handler import AceStepHandler
        from acestep.inference import GenerationConfig, GenerationParams, generate_music
        from acestep.llm_inference import LLMHandler

    
        global STAGED_FREE
        STAGED_FREE = bool(args.staged_free)
        wrap_class_stages(sampler)

        # ---- initialise DiT / condition encoder / text encoder / VAE (+ MLX backends)
        dit = AceStepHandler()
        init_start = time.monotonic()
        status, ok_init = dit.initialize_service(
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
        init_wall = time.monotonic() - init_start
        result["init"] = {
            "ok": bool(ok_init),
            "status": status,
            "wall_s": round(init_wall, 2),
            "peak_footprint_mb": round(sampler.max_fp / 2**20, 1),
            "torch_model": _module_bytes_torch(dit.model) if dit.model is not None else {},
            "torch_vae": _module_bytes_torch(dit.vae) if dit.vae is not None else {},
            "torch_text_encoder": _module_bytes_torch(dit.text_encoder) if dit.text_encoder is not None else {},
            "mlx_dit": _module_bytes_mlx(dit.mlx_decoder) if getattr(dit, "mlx_decoder", None) is not None else {},
            "mlx_vae": _module_bytes_mlx(dit.mlx_vae) if getattr(dit, "mlx_vae", None) is not None else {},
            "use_mlx_dit": bool(getattr(dit, "use_mlx_dit", False)),
            "use_mlx_vae": bool(getattr(dit, "use_mlx_vae", False)),
            "dtype": str(dit.dtype),
        }
        if not ok_init:
            raise RuntimeError(f"initialize_service failed: {status}")

        wrap_instance_stages(sampler, dit.model)

        lm_handler = LLMHandler()
        if args.lm != "none":
            lm_name = f"acestep-5Hz-lm-{args.lm}"
            lm_status, lm_ok = lm_handler.initialize(
                checkpoint_dir=CKPT,
                lm_model_path=lm_name,
                backend="mlx",
                device="mps",
                offload_to_cpu=False,
                dtype=None,
            )
            result["lm_init"] = {"ok": bool(lm_ok), "status": lm_status, "model": lm_name}
            if not lm_ok:
                raise RuntimeError(f"LM initialize failed: {lm_status}")

        # ---- generation params
        if args.lm == "none" or THINKING is False:
            thinking = False if THINKING is None else THINKING
        else:
            thinking = True
        params = GenerationParams(
            task_type="text2music",
            caption=caption,
            lyrics=lyrics if not instrumental else "[Instrumental]",
            instrumental=instrumental,
            vocal_language=vocal_language,
            duration=args.duration,
            inference_steps=args.steps,
            seed=args.seed,
            shift=args.shift,
            infer_method="ode",
            guidance_scale=1.0 if args.steps <= 8 else 7.0,
            thinking=thinking,
            use_cot_metas=thinking,
            use_cot_caption=thinking,
            use_cot_language=thinking,
            use_constrained_decoding=thinking,
            lm_negative_prompt="NO USER INPUT",
        )
        batch = max(1, int(args.batch))
        config = GenerationConfig(
            batch_size=batch,
            use_random_seed=False,
            seeds=[args.seed + i for i in range(batch)],
            audio_format="wav",
        )
        result["params"] = {
            "thinking": thinking,
            "inference_steps": params.inference_steps,
            "guidance_scale": params.guidance_scale,
            "shift": params.shift,
            "infer_method": params.infer_method,
            "use_adg": params.use_adg,
            "dcw_enabled": params.dcw_enabled,
            "audio_format": config.audio_format,
        }

        save_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "audio", args.tag or "run")
        os.makedirs(save_dir, exist_ok=True)

        try:
            mx.reset_peak_memory()
        except Exception:
            pass
        try:
            torch.mps.reset_peak_memory()
        except Exception:
            pass

        gen_start = time.monotonic()
        gen_result = generate_music(dit, lm_handler, params, config, save_dir=save_dir)
        gen_wall = time.monotonic() - gen_start

        result["wall_s"] = round(gen_wall, 2)
        result["result"] = {
            "success": bool(getattr(gen_result, "success", False)),
            "status_message": getattr(gen_result, "status_message", ""),
            "error": getattr(gen_result, "error", ""),
        }

        extra = getattr(gen_result, "extra_outputs", {}) or {}
        tc = extra.get("time_costs", {}) or {}
        result["time_costs"] = {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in tc.items()}
        result["extra_keys"] = sorted(list(extra.keys()))
        if "lm_metadata" in extra and isinstance(extra["lm_metadata"], list) and extra["lm_metadata"]:
            md = extra["lm_metadata"][0]
            result["lm_metadata"] = md if isinstance(md, dict) else str(md)
        elif isinstance(extra.get("lm_metadata"), dict):
            result["lm_metadata"] = extra["lm_metadata"]

        for i, audio in enumerate(getattr(gen_result, "audios", []) or []):
            src = audio.get("path")
            info = {
                "index": i,
                "orig_path": src,
                "key": audio.get("key"),
                "params": {k: v for k, v in (audio.get("params") or {}).items() if k != "lyrics"},
            }
            if src and os.path.exists(src):
                ext = os.path.splitext(src)[1]
                dst = os.path.join(args.wav_dir, f"{args.tag or 'run'}_{i}{ext}")
                try:
                    import shutil

                    shutil.copy2(src, dst)
                    info["saved"] = dst
                except Exception as exc:  # noqa: BLE001
                    info["save_error"] = repr(exc)
            result["wavs"].append(info)

        result["ok"] = bool(result["result"]["success"])

        # ---- per-stage rollup
        stage_names = sorted({n for (n, k, _, _) in sampler.events if k == "window"})
        for name in stage_names:
            result["stages"][name] = sampler.stage_peak(name)
        result["process_peak_footprint_mb"] = round(sampler.max_fp / 2**20, 1)
        result["mlx_mem_end"] = _mlx_mem()
        result["torch_mem_end"] = _torch_mem()
        try:
            import resource

            result["ru_maxrss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)
        except Exception:
            pass
        result["marks"] = [{"t": round(t, 2), "what": w} for t, w in sampler.marks]

        return 0

    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], file=sys.stderr)
        return 1
    finally:
        sampler.stop()
        result["process_peak_footprint_mb"] = round(max(sampler.max_fp, result.get("process_peak_footprint_mb", 0) or 0) / 2**20, 1)
        stage_names = sorted({n for (n, k, _, _) in sampler.events if k == "window"})
        for name in stage_names:
            if name not in result["stages"]:
                result["stages"][name] = sampler.stage_peak(name)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        log_path = os.path.splitext(args.out)[0] + ".log.json"
        with open(log_path, "w") as f:
            json.dump(LOGS, f, indent=1, default=str)
        print(json.dumps({k: result.get(k) for k in ("ok", "error", "wall_s", "process_peak_footprint_mb", "time_costs")}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
