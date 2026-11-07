### bf16 / lm=0.6B

| dur | init wall | text enc s/peak | condition s/peak | DiT 8 steps s/peak | VAE decode s/peak | gen wall (model) | process peak | MLX peak | torch@exit | (all memory in GiB) |
|---:|---:|---|---|---|---|---:|---:|---:|---:|
| 30 s | 7.8 s | 0.28 s / 10.97 GiB | 0.38 s / 11.13 GiB | 3.13 s / 12.72 GiB | 4.46 s / 16.80 GiB | 16.1 s | 16.80 GiB | 7.1 GiB | 7.7 GiB |
| 120 s | 7.2 s | 0.26 s / 11.20 GiB | 0.38 s / 11.36 GiB | 12.09 s / 13.18 GiB | 12.06 s / 16.82 GiB | 40.8 s | 16.82 GiB | 7.2 GiB | 7.7 GiB |
| 240 s | 7.6 s | 0.22 s / 11.59 GiB | 0.39 s / 11.75 GiB | 23.95 s / 13.62 GiB | 21.62 s / 16.87 GiB | 79.2 s | 16.87 GiB | 7.2 GiB | 7.8 GiB |

### bf16 / lm=none

| dur | init wall | text enc s/peak | condition s/peak | DiT 8 steps s/peak | VAE decode s/peak | gen wall (model) | process peak | MLX peak | torch@exit | (all memory in GiB) |
|---:|---:|---|---|---|---|---:|---:|---:|---:|
| 30 s | 7.9 s | 0.09 s / 10.63 GiB | 0.88 s / 10.78 GiB | 3.02 s / 12.28 GiB | 3.70 s / 15.99 GiB | 8.7 s | 15.99 GiB | 5.9 GiB | 6.4 GiB |
| 120 s | 13.2 s | 0.08 s / 10.79 GiB | 0.49 s / 10.90 GiB | 10.33 s / 12.75 GiB | 11.40 s / 16.45 GiB | 24.3 s | 16.45 GiB | 5.9 GiB | 6.5 GiB |
| 240 s | 7.2 s | 0.08 s / 10.79 GiB | 1.02 s / 10.88 GiB | 22.39 s / 12.78 GiB | 21.49 s / 16.60 GiB | 49.8 s | 16.60 GiB | 6.0 GiB | 6.6 GiB |

### bf16-4bit / lm=none

| dur | init wall | text enc s/peak | condition s/peak | DiT 8 steps s/peak | VAE decode s/peak | gen wall (model) | process peak | MLX peak | torch@exit | (all memory in GiB) |
|---:|---:|---|---|---|---|---:|---:|---:|---:|
| 30 s | 9.0 s | 0.07 s / 11.38 GiB | 1.05 s / 11.53 GiB | 2.70 s / 12.22 GiB | 3.84 s / 16.28 GiB | 8.6 s | 16.28 GiB | 3.8 GiB | 4.3 GiB |
| 120 s | 7.5 s | 0.08 s / 11.53 GiB | 0.60 s / 11.67 GiB | 9.98 s / 13.43 GiB | 11.67 s / 16.45 GiB | 24.1 s | 16.45 GiB | 3.8 GiB | 4.4 GiB |
| 240 s | 8.8 s | 0.09 s / 11.55 GiB | 0.67 s / 11.71 GiB | 22.72 s / 13.42 GiB | 21.64 s / 16.62 GiB | 49.0 s | 16.62 GiB | 3.9 GiB | 4.5 GiB |
