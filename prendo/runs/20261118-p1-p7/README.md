# P1/P2/P4/P5/P7 experiment run (2026-11-18)

HPC: Imperial CX3, queue `v1_gpu72` (L40S), user `yl15425`.
Training data: `VRCaps_lumen_100_v2` (88 train / 12 test), unchanged.
P1/P1+P2 use an isolated patched `depth-diff-gaussian-rasterization` build
(`auxiliary.h: p_view.z <= 0.2f -> 0.01f`) at
`$HOME/prendo/envs/nearcull`. P5 and the baseline use the original 0.2 build.

## Runs

| Job | ID | What |
|---|---|---|
| near-cull build | 3025961 | compiled patched rasterizer; `cull_threshold=0.01` |
| P5 prune trace | 3025783 | original 0.2 cull, f0=0.05, 30k, checkpoint/PLY every 1000, prune logs |
| P1 | 3025996 | 0.01 cull, f0=0.05, 30k |
| P1+P2 | 3025997 | 0.01 cull, f0_init=0.03, 30k |
| P1 render non-L2 | 3025998 | 391 entries, patched rasterizer |
| P1 render L2 | 3025999 | 220 current L2 entries, n=2.400499, patched rasterizer |
| P4 diagnostics | 3026000 | baseline (0.2) + P1/P1+P2 (0.01), 49 L0 views |

## 611-entry protocol

23 configs = 16 B1 non-L2 (391 valid) + 7 current L2 directions (220 valid).
Old 9 L2 configs (60 deg, n=23.135) and `L2_yaw±40` are excluded.
Staged root:
`/rds/general/user/yl15425/ephemeral/prendo/data/staged_p1_611/VRCaps_lumen_100_v2`.

## Success criteria

- P1: L0 ~42 dB; near-wall (GT z <= 0.2) gap to far <= 2 dB; L3 near-wall median error <=10%.
- P2: F0 not spatially constant; L0 top 0.1% closer to Unity 0.298; P1+P2 vs P1 isolates P2.
- P4: baseline depth in cull region ~1.7x GT; P1 closer to 1.0x; P1+P2 specular larger.
- P5: near-wall Gaussians disappear after the 3000-step opacity reset, at the 4000-step prune.
- P7: learned local attenuation exponent closer to Unity 2 after P1; else formula limitation.

Results will be appended to `EVALUATION.md`.
