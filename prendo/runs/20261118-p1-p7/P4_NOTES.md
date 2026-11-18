# P4 GPU diagnostics (49 L0 views)

Baseline uses the original 0.2 rasterizer; P1/P1+P2 use the 0.01 build.

| model | depth/GT median, near (GT z<=0.2) | far | L0 top 0.1% | specular top 0.1% | specular nonzero |
|---|---:|---:|---:|---:|---:|
| baseline | 1.596 | 0.992 | 0.998 | 0.993 | 11.7% |
| P1 | 0.985 | 0.991 | 1.000 | 0.997 | 6.7% |
| P1+P2 | 0.985 | 0.991 | 1.000 | 0.992 | 6.6% |

The baseline near-wall depth ratio (1.60) is close to the CPU approximation
reported in B04/B10 (1.7), validating the CPU analysis. P1 removes the error
(0.985). P2's F0 variation does not increase the pure-specular highlight:
P1+P2's top 0.1% is 0.992 vs P1 0.997 and baseline 0.993; the nonzero
fraction is also unchanged. L0 top 0.1% is clipped at 1.0 for P1/P1+P2 and
cannot separate the two.
