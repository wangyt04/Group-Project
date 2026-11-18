# P1 / P1+P2 training notes

Both runs use the patched 0.01 near-cull rasterizer
(`$HOME/prendo/envs/nearcull`), 30k steps, 4 CPUs / 1 L40S.

| iter | P1 test PSNR | P1+P2 test PSNR | P1 points | P1+P2 points |
|---:|---:|---:|---:|---:|
| 1,000 | 36.71 | 34.70 | 300,000 | 300,000 |
| 10,000 | 39.66 | 39.51 | 270,724 | 272,940 |
| 20,000 | 41.19 | 40.99 | 270,724 | 272,940 |
| 30,000 | 41.68 | 41.23 | 270,724 | 272,940 |

Baseline (original 0.2 cull) test PSNR at 30k: 33.64 dB, 213,091 Gaussians.
At 10k P1 is already +6.0 dB over the baseline's 30k score, and the point
count is ~270k instead of the baseline's ~214k, consistent with near-wall
Gaussians surviving. P1+P2 tracks P1 closely at 10k.

The 611-entry L0/L3/L2 evaluation is pending the P1 render jobs.

Final 30k test PSNR: P1 **41.68 dB**, P1+P2 **41.23 dB**, baseline **33.64 dB**.
P1 has 270,724 Gaussians vs baseline 213,091.
