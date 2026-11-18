# P5 prune-timeline result

Run `vrcaps_lumen_100_v2_p5_prune_r30k`, original 0.2 cull and f0=0.05,
30k steps, PLY/checkpoint every 1000. Near-wall defined by projecting points
into frame 0010 and keeping `0 < z <= 0.2`.

| iter | points | near-wall | near-wall frac |
|---:|---:|---:|---:|
| 0 | 300,000 | 35,432 | 11.81% |
| 1,000 | 300,000 | 35,457 | 11.82% |
| 2,000 | 300,000 | 35,471 | 11.82% |
| 3,000 | 274,172 | 34,444 | 12.56% |
| 4,000 | 271,744 | 34,405 | 12.66% |
| 5,000 | 230,328 | 1,721 | 0.75% |
| 6,000 | 226,749 | 1,571 | 0.69% |
| 10,000 | 214,237 | 1,206 | 0.56% |
| 30,000 | 214,237 | 1,196 | 0.56% |

The iteration-4000 snapshot is written before that iteration's
densification/prune. The prune event at 4000 removes 42,596 points, of which
**32,706 are near-wall**; the 5000 snapshot then shows 1,721. The 3000
opacity reset touches 271,744 points (34,410 near-wall) but does not remove
them.

Prune removals of near-wall points by reason over the run:
`densify_prune` 34,571, `opacity_reset` 37,064 (reset, not removal), `split` 310.

Conclusion: P5 confirms C1's mechanism and timing: the 3000-step opacity reset
is followed by the 4000-step prune, which deletes ~95% of the near-wall
Gaussians; about 1.2k survive and remain stable through 30k.
