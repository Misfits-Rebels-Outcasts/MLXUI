#!/usr/bin/env python3
"""MC5-3b (R6): convert ACE-Step's silence_latent.pt into the raw f32 resource the app bundles.

Input : acestep-v15-turbo/silence_latent.pt from ACE-Step/Ace-Step1.5 @ 19671f4 (torch.load, weights_only=True,
        float32 tensor [1, 64, 15000]).
Output: little-endian f32, row-major [15000, 64] (the upstream load-time transpose of the input), 3,840,000 bytes.
Prints the input and output sha256. Needs torch; run it from the pinned oracle venv. Not part of the app bundle.

    python3 -I Tools/AceStep/convert_silence_latent.py <silence_latent.pt> MLXUI/Resources/AceStep/silence_latent_f32.bin
"""
import hashlib, sys

EXPECTED_IN_SHA256 = "a778e9dd942f5e8b2c09c55370782d318834432b03dabbcdf70e6ed49ad6358b"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    got = sha256(src)
    if got != EXPECTED_IN_SHA256:
        sys.exit(f"input sha256 {got} != pinned {EXPECTED_IN_SHA256}")
    import torch
    t = torch.load(src, map_location="cpu", weights_only=True)
    assert t.dtype == torch.float32 and tuple(t.shape) == (1, 64, 15000), (t.dtype, tuple(t.shape))
    out = t.transpose(1, 2).contiguous()[0]          # [15000, 64]
    assert tuple(out.shape) == (15000, 64)
    data = out.numpy().astype("<f4").tobytes()
    assert len(data) == 15000 * 64 * 4
    with open(dst, "wb") as f:
        f.write(data)
    print("input  sha256", got)
    print("output sha256", sha256(dst), len(data), "bytes")


if __name__ == "__main__":
    main()
