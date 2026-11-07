### bf16 / lm=0.6B

| dur | init wall | text enc s/peak | condition s/peak | DiT 8 steps s/peak | VAE decode s/peak | gen wall (model) | process peak | MLX peak | torch@exit |
|---:|---:|---|---|---|---|---:|---:|---:|---:|
| 30 s | 7.8 s | 0.28 s / 10.97 GB | 0.38 s / 11.13 GB | 3.13 s / 12.72 GB | 4.46 s / 16.80 GB | 16.1 s | 16.80 GB | 7.1 GB | 7.7 GB |
| 120 s | 7.2 s | 0.26 s / 11.20 GB | 0.38 s / 11.36 GB | 12.09 s / 13.18 GB | 12.06 s / 16.82 GB | 40.8 s | 16.82 GB | 7.2 GB | 7.7 GB |
| 240 s | 7.6 s | 0.22 s / 11.59 GB | 0.39 s / 11.75 GB | 23.95 s / 13.62 GB | 21.62 s / 16.87 GB | 79.2 s | 16.87 GB | 7.2 GB | 7.8 GB |

### bf16 / lm=none

| dur | init wall | text enc s/peak | condition s/peak | DiT 8 steps s/peak | VAE decode s/peak | gen wall (model) | process peak | MLX peak | torch@exit |
|---:|---:|---|---|---|---|---:|---:|---:|---:|
| 30 s | 7.9 s | 0.09 s / 10.63 GB | 0.88 s / 10.78 GB | 3.02 s / 12.28 GB | 3.70 s / 15.99 GB | 8.7 s | 15.99 GB | 5.9 GB | 6.4 GB |
| 120 s | 13.2 s | 0.08 s / 10.79 GB | 0.49 s / 10.90 GB | 10.33 s / 12.75 GB | 11.40 s / 16.45 GB | 24.3 s | 16.45 GB | 5.9 GB | 6.5 GB |
| 240 s | 7.2 s | 0.08 s / 10.79 GB | 1.02 s / 10.88 GB | 22.39 s / 12.78 GB | 21.49 s / 16.60 GB | 49.8 s | 16.60 GB | 6.0 GB | 6.6 GB |

### bf16-4bit / lm=none

| dur | init wall | text enc s/peak | condition s/peak | DiT 8 steps s/peak | VAE decode s/peak | gen wall (model) | process peak | MLX peak | torch@exit |
|---:|---:|---|---|---|---|---:|---:|---:|---:|
| 30 s | 9.0 s | 0.07 s / 11.38 GB | 1.05 s / 11.53 GB | 2.70 s / 12.22 GB | 3.84 s / 16.28 GB | 8.6 s | 16.28 GB | 3.8 GB | 4.3 GB |
| 120 s | 7.5 s | 0.08 s / 11.53 GB | 0.60 s / 11.67 GB | 9.98 s / 13.43 GB | 11.67 s / 16.45 GB | 24.1 s | 16.45 GB | 3.8 GB | 4.4 GB |
| 240 s | 8.8 s | 0.09 s / 11.55 GB | 0.67 s / 11.71 GB | 22.72 s / 13.42 GB | 21.64 s / 16.62 GB | 49.0 s | 16.62 GB | 3.9 GB | 4.5 GB |
