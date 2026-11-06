"""One-time generator for MLXUITests/Fixtures/DepthAnything (MC3-3).

Runs the Python reference — `transformers`' `DepthAnythingForDepthEstimation` + `DPTImageProcessor` — on
one deterministic synthetic scene and on the repo's `sample-image.png`, and writes the reference tensors,
the final 8-bit PNGs and the preprocessing size table as raw little-endian files.

Usage (never run from a committed environment):
    python3 -I make_depth_golden.py <weights_dir> <out_dir> <repo_root>

`weights_dir` is the pinned `depth-anything/Depth-Anything-V2-Small-hf` snapshot
(revision 5426e4f0f36572d16453bbda7a8389317b1bef99): config.json, preprocessor_config.json,
model.safetensors. `repo_root` is the MLXUI checkout (for MLXUI/Resources/Samples/sample-image.png).

Layout convention for every `.f32` activation file: NHWC (`[1, H, W, C]`) little-endian float32, i.e. the
PyTorch NCHW tensor transposed so a Swift/MLX reader can reshape it directly. `predicted_depth` is
H x W float32 little-endian. The PNGs are 8-bit grayscale, near = bright (decision 4: bicubic upsample of the
float prediction to the input's pixel size, then per-image min–max normalisation, then quantisation).
"""

import hashlib
import json
import sys

import numpy as np
import torch
from PIL import Image, ImageOps

import transformers

torch.set_grad_enabled(False)

weights_dir = sys.argv[1]
out = sys.argv[2]
repo_root = sys.argv[3]

from transformers import DPTImageProcessor, DepthAnythingForDepthEstimation  # noqa: E402

processor = DPTImageProcessor.from_pretrained(weights_dir)
model = DepthAnythingForDepthEstimation.from_pretrained(weights_dir).eval()

PATCH = model.config.patch_size
REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"


def nhwc(tensor):
    """[B, C, H, W] -> [B, H, W, C] float32, or [B, H, W] unchanged."""
    array = tensor.detach().cpu().numpy().astype(np.float32)
    if array.ndim == 4:
        return array.transpose(0, 2, 3, 1).copy()
    return array.copy()


def write_f32(name, array):
    array.astype("<f4").tofile(out + "/" + name)
    return list(array.shape)


def synthetic_scene():
    """A near square over a far gradient, 196 x 140 (W x H), deterministic, generated here.

    The square sits in the upper right; the background is a smooth vertical gradient. Both are grey
    (near-neutral) so the depth cue is shape, not colour.
    """
    height, width = 140, 196
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    background = 40.0 + 70.0 * (yy / (height - 1))
    square = (((xx - 130.0) ** 2 + (yy - 55.0) ** 2) < 38.0**2) | (
        (xx > 92.0) & (xx < 168.0) & (yy > 17.0) & (yy < 93.0)
    )
    grey = np.where(square, 205.0, background)
    rgb = np.stack([grey, grey, grey], axis=-1)
    return np.clip(rgb, 0, 255).round().astype(np.uint8)


def run_reference(pixel_values):
    """The reference forward, with each stage's output returned explicitly.

    Returns (features, reassembled, fusion, predicted_depth). `features` are the backbone feature maps
    after its layer norm (still [B, seq, C] with the CLS token); `reassembled` are the four reassemble
    outputs (NCHW); `fusion` is the neck's final fused map; `predicted_depth` is [1, H, W].
    """
    backbone_out = model.backbone(pixel_values)
    feature_maps = backbone_out.feature_maps
    _, _, height, width = pixel_values.shape
    patch_height, patch_width = height // PATCH, width // PATCH
    reassembled = model.neck.reassemble_stage(list(feature_maps), patch_height, patch_width)
    neck_features = [model.neck.convs[i](feature) for i, feature in enumerate(reassembled)]
    fused = model.neck.fusion_stage(neck_features)
    predicted_depth = model.head(fused, patch_height, patch_width)
    return feature_maps, reassembled, fused[-1], predicted_depth


def final_png(predicted_depth, target_height, target_width):
    """Decision 4: bicubic upsample of the float prediction to the input's pixel size, per-image
    min–max normalisation, then 8-bit quantisation. A constant prediction is all black, never NaN."""
    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(0).unsqueeze(1),
        size=(target_height, target_width),
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    low, high = depth.min(), depth.max()
    if float(high - low) == 0.0:
        grey = torch.zeros_like(depth)
    else:
        grey = ((depth - low) / (high - low) * 255.0).round().clamp(0, 255)
    return grey.to(torch.uint8).cpu().numpy()


def save_png(name, grey):
    Image.fromarray(grey, mode="L").save(out + "/" + name)


meta = {
    "reference": "transformers DepthAnythingForDepthEstimation + DPTImageProcessor",
    "transformers": transformers.__version__,
    "torch": torch.__version__,
    "numpy": np.__version__,
    "hf_repository": "depth-anything/Depth-Anything-V2-Small-hf",
    "hf_revision": REVISION,
    "weights_sha256": hashlib.sha256(open(weights_dir + "/model.safetensors", "rb").read()).hexdigest(),
    "layout": "NHWC [1,H,W,C] float32 little-endian for activations; predicted_depth H x W; PNG 8-bit grey",
    "preprocessing": {
        "size": {"height": 518, "width": 518},
        "keep_aspect_ratio": True,
        "ensure_multiple_of": 14,
        "resample": "bicubic",
        "rescale_factor": 1.0 / 255.0,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "constrain_rule": "round(value/multiple)*multiple (Python round, half-to-even)",
    },
    "preprocessing_tolerance": {
        "note": "Swift CoreGraphics .high resize vs PIL bicubic; differences are localized at sharp edges "
                "(median ~0), so preprocessing is compared separately from model parity",
        "measured_on": "a, 140x140 -> 364x518: median 0, p99 0.16, max 0.47 (normalized units)",
        "median_abs_delta": 0.01,
        "p99_abs_delta": 0.25,
        "max_abs_delta": 0.6,
    },
    "final_png_rule": "bicubic(F.interpolate, align_corners=False) to the input's pixel size, then min-max to 0..255",
    "cases": {},
}

# ---- (a) the synthetic scene, full model path ---------------------------------------------------
scene = synthetic_scene()
scene.tofile(out + "/a_input.rgb")
scene_image = Image.fromarray(scene)

a_inputs = processor(images=scene_image, return_tensors="pt")
a_pixel = a_inputs["pixel_values"]
a_features, a_reassembled, a_fusion, a_predicted = run_reference(a_pixel)
a_final = final_png(a_predicted[0], scene.shape[0], scene.shape[1])
save_png("a_final.png", a_final)

# Cross-check the explicit stage calls against the model's own forward.
with_grad = model(pixel_values=a_pixel)
assert torch.allclose(with_grad.predicted_depth, a_predicted, atol=0, rtol=0) or torch.allclose(
    with_grad.predicted_depth, a_predicted
), "explicit forward diverged from model()"

# Layer-by-layer parity uses the smaller non-square probe below (see its comment): at the model's real
# preprocessed size a single fusion map is 208 x 296 x 64 = 15.8 MB and the four feature maps are 5.9 MB,
# which would put the fixture folder far past the issue's ~4 MB guidance. (a) keeps full-size end-to-end
# evidence (the preprocessed tensor, predicted_depth and the final PNG).
a_files = {
    "preprocessed": {"file": "a_preprocessed.f32", "shape": write_f32("a_preprocessed.f32", nhwc(a_pixel))},
    "predicted_depth": {
        "file": "a_predicted_depth.f32",
        "shape": write_f32("a_predicted_depth.f32", a_predicted[0].detach().cpu().numpy().astype(np.float32)),
    },
    "final_png": {"file": "a_final.png", "height": int(scene.shape[0]), "width": int(scene.shape[1])},
    "input": {"file": "a_input.rgb", "height": int(scene.shape[0]), "width": int(scene.shape[1])},
}
meta["cases"]["a"] = a_files

# ---- the layer-parity probe: the same scene through the reference processor at a smaller size -----
# Keeping the full-size stage tensors (features are [1, 963, 384] each; a fusion map is 208 x 296 x 64)
# would put the fixture folder far past the issue's ~4 MB guidance. Layer parity runs on a smaller
# non-square processed tensor (112 x 154, patch grid 8 x 11): it exercises the same position-embedding
# interpolation and every layer, while (a) and (b) keep full-size end-to-end parity.
probe_inputs = processor(images=scene_image, return_tensors="pt", size={"height": 112, "width": 112})
probe_pixel = probe_inputs["pixel_values"]
p_features, p_reassembled, p_fusion, p_predicted = run_reference(probe_pixel)
probe_files = {
    "preprocessed": {"file": "probe_preprocessed.f32", "shape": write_f32("probe_preprocessed.f32", nhwc(probe_pixel))},
    "features": [
        {"file": f"probe_feature_{i}.f32", "shape": write_f32(f"probe_feature_{i}.f32", nhwc(f))}
        for i, f in enumerate(p_features)
    ],
    "reassemble": [
        {"file": f"probe_reassemble_{i}.f32", "shape": write_f32(f"probe_reassemble_{i}.f32", nhwc(f))}
        for i, f in enumerate(p_reassembled)
    ],
    "fusion": {"file": "probe_fusion.f32", "shape": write_f32("probe_fusion.f32", nhwc(p_fusion))},
    "predicted_depth": {
        "file": "probe_predicted_depth.f32",
        "shape": write_f32("probe_predicted_depth.f32", p_predicted[0].detach().cpu().numpy().astype(np.float32)),
    },
}
meta["cases"]["probe"] = probe_files

# ---- (b) the repo's sample photo, full model path -------------------------------------------------
sample = ImageOps.exif_transpose(Image.open(repo_root + "/MLXUI/Resources/Samples/sample-image.png")).convert("RGB")
b_inputs = processor(images=sample, return_tensors="pt")
b_pixel = b_inputs["pixel_values"]
_, _, _, b_predicted = run_reference(b_pixel)
b_final = final_png(b_predicted[0], sample.height, sample.width)
save_png("b_final.png", b_final)
meta["cases"]["b"] = {
    "preprocessed": {"file": "b_preprocessed.f32", "shape": write_f32("b_preprocessed.f32", nhwc(b_pixel))},
    "predicted_depth": {
        "file": "b_predicted_depth.f32",
        "shape": write_f32("b_predicted_depth.f32", b_predicted[0].detach().cpu().numpy().astype(np.float32)),
    },
    "final_png": {"file": "b_final.png", "height": int(sample.height), "width": int(sample.width)},
    "fed_as": "PIL Image.open then ImageOps.exif_transpose then convert('RGB'); no other colour transform",
}

# ---- Trap 2: the preprocessing output-size table ------------------------------------------------
size_table = []
for height, width in [(140, 196), (196, 140), (518, 518), (30, 40), (200, 600)]:
    image = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    processed = processor(images=image, return_tensors="pt")["pixel_values"]
    size_table.append({
        "input_height": height,
        "input_width": width,
        "output_height": int(processed.shape[2]),
        "output_width": int(processed.shape[3]),
    })
meta["size_table"] = size_table

json.dump(meta, open(out + "/depth-golden.json", "w"), indent=2, sort_keys=True)
print(json.dumps(meta, indent=2, sort_keys=True))
