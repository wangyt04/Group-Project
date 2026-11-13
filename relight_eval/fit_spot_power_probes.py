"""Fit PR-ENDO's spotlight exponent n to a matched Point/Spot calibration set, using the
supplementary angular probes as well as the forward pair.

    python fit_spot_power_probes.py <calibration_root> [--out <dir>]

The calibration root must contain, for every frame and config:
    relight/gt/L0/noshadow/<frame>.exr          auxiliary point reference
    relight/gt/<spot_cfg>/noshadow/<frame>.exr  matched spot
    relight/depth/<frame>.exr                   depth, invalid pixels 0

L2 and the Point are rendered at the same position and intensity, so their per-pixel
ratio depends only on the angle between the pixel ray and the beam axis.  This script
fits `ratio ~ cos(theta)^n` on:
  - the forward pair alone (angles up to the camera corner, ~52.4 deg),
  - the +/-20 yaw/pitch probes (angles up to ~69 deg),
  - all of them together (the value written for rendering).

The new HDRP profile has a flat inner core (inner half-angle 10 deg) and an outer
rolloff to 80 deg.  cos^n cannot represent the flat core exactly, so the fit is an
approximation; the script reports the residuals and the measured profile.  The old
n from the 60-degree cone delivery is invalid for this profile.

Writes calibration.json and spot_fit.png to --out (default eval_relight/<dataset>).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

import relight_io as io

FRAMES_DEFAULT = ["0001", "0032", "0050", "0067", "0097"]


def beam_angle_deg(meta, c2w, light_dir, w, h):
    """Angle in degrees between each pixel's ray and the spot beam axis."""
    right, up, forward, _ = io.camera_axes(c2w)
    x, y = io.pixel_grid(meta, w, h)
    rays = np.stack(
        [forward[0] + x * right[0] + y * up[0],
         forward[1] + x * right[1] + y * up[1],
         forward[2] + x * right[2] + y * up[2]], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    d = np.asarray(light_dir, dtype=np.float64)
    d /= np.linalg.norm(d)
    return np.degrees(np.arccos(np.clip(rays @ d, -1.0, 1.0)))


def collect(ds, cfg, frames):
    """(angle_deg, ratio) pairs for one spot config over the given frames."""
    entries = {(e["config"], str(e["frame"]).zfill(4)): e for e in ds.entries}
    alphas, ratios = [], []
    for frame in frames:
        e = entries.get((cfg, frame))
        if e is None:
            continue
        g0 = io.read_exr_rgb(ds.root / "relight/gt/L0/noshadow" / f"{frame}.exr")
        gs = io.read_exr_rgb(ds.root / "relight/gt" / cfg / "noshadow" / f"{frame}.exr")
        depth = io.read_exr_depth(ds.root / "relight/depth" / f"{frame}.exr")
        h, w = g0.shape[:2]
        ang = beam_angle_deg(ds.relight_meta, ds.relight_frames[frame]["transform_matrix"],
                             e["light_dir_world"], w, h)
        y0, ys = io.luminance(g0), io.luminance(gs)
        m = (depth > 0) & (y0 > 0.02) & np.isfinite(y0) & np.isfinite(ys)
        alphas.append(ang[m])
        ratios.append((ys / np.maximum(y0, 1e-9))[m])
    if not alphas:
        raise SystemExit(f"no pixels collected for {cfg}")
    return np.concatenate(alphas), np.concatenate(ratios)


def fit(alpha_deg, ratio):
    """Least-squares n for ratio ~ cos(theta)^n."""
    cos = np.cos(np.radians(alpha_deg))
    loss = lambda log_n: float(np.mean((cos ** np.exp(log_n) - ratio) ** 2))
    best = minimize_scalar(loss, bounds=(np.log(1e-3), np.log(1000.0)),
                           method="bounded", options={"xatol": 1e-6})
    return float(np.exp(best.x)), float(np.sqrt(best.fun))


def profile(alpha_deg, ratio, weighted, step_deg=1.0):
    """Per-degree medians; `weighted` is the combined angle/ratio for the median."""
    rows = []
    for lo in np.arange(0, 71, step_deg):
        m = (alpha_deg >= lo) & (alpha_deg < lo + step_deg)
        if m.sum() >= 20:
            rows.append([round(lo + step_deg / 2, 2), float(np.median(ratio[m])), int(m.sum())])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--forward", default="L2_fwd")
    ap.add_argument("--probe-prefix", default="CAL_")
    ap.add_argument("--frames", nargs="*", default=FRAMES_DEFAULT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ds = io.Dataset(args.dataset)
    probe_ids = [c["id"] for c in ds.lights["configs"]
                 if c["id"].startswith(args.probe_prefix) and c["type"] == "spot"]
    if not probe_ids:
        raise SystemExit(f"no probe configs starting with {args.probe_prefix!r}")

    fwd_a, fwd_r = collect(ds, args.forward, args.frames)
    probe_a, probe_r = [], []
    for cid in probe_ids:
        a, r = collect(ds, cid, args.frames)
        probe_a.append(a)
        probe_r.append(r)
    probe_a, probe_r = np.concatenate(probe_a), np.concatenate(probe_r)
    all_a, all_r = np.concatenate([fwd_a, probe_a]), np.concatenate([fwd_r, probe_r])

    n_fwd, rm_fwd = fit(fwd_a, fwd_r)
    n_probe, rm_probe = fit(probe_a, probe_r)
    n_all, rm_all = fit(all_a, all_r)

    cfg = ds.configs[args.forward]
    out = args.out or (Path("eval_relight") / ds.name)
    out.mkdir(parents=True, exist_ok=True)
    calib = {
        "dataset": ds.name,
        "source_zip": f"{ds.name}.zip",
        "spot_power": n_all,
        "spot_fit_rmse": rm_all,
        "spot_power_forward_only": n_fwd,
        "spot_fit_rmse_forward_only": rm_fwd,
        "spot_power_probes_only": n_probe,
        "spot_fit_rmse_probes_only": rm_probe,
        "frames": args.frames,
        "n_pixels": int(all_a.size),
        "spot_inner_deg": cfg.get("spot_inner_deg"),
        "spot_outer_deg": cfg.get("spot_outer_deg"),
        "spot_inner_full_deg": cfg.get("unity_full_inner_angle_deg"),
        "spot_outer_full_deg": cfg.get("unity_full_spot_angle_deg"),
        "innerSpotPercent": cfg.get("HDRP_innerSpotPercent"),
        "fit_inputs": "L2/point ratio vs angle from beam axis; forward pair plus all "
                      f"{args.probe_prefix}* probes ({len(args.frames)} frames each)",
        "spot_profile_deg": profile(all_a, all_r, None),
        "notes": "cos^n approximates the HDRP flat core plus outer rolloff; old n from the "
                 "60-degree cone is invalid for this profile. spot_power is the probe-aware "
                 "fit used for rendering; forward-only and probe-only variants are reported "
                 "for transparency.",
    }
    (out / "calibration.json").write_text(json.dumps(calib, indent=2) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for a, r, style, label in [
        (fwd_a, fwd_r, dict(marker="o", ms=3, color="0.35", ls="none"), "Unity forward L2/L0 (median per 1 deg)"),
        (probe_a, probe_r, dict(marker="s", ms=2.5, color="C1", alpha=0.7, ls="none"), "Unity probes +/-20 (median per 1 deg)"),
    ]:
        rows = profile(a, r, None)
        ax.plot([x[0] for x in rows], [x[1] for x in rows], label=label, **style)
    aa = np.radians(np.linspace(0, 70, 300))
    ax.plot(np.degrees(aa), np.cos(aa) ** n_all, color="C0",
            label=f"cos^n, n={n_all:.4f} (used)")
    for x in (cfg.get("spot_inner_deg"), cfg.get("spot_outer_deg")):
        if x:
            ax.axvline(x, color="0.7", lw=0.8, ls=":")
    ax.set(xlabel="angle from beam axis (deg)", ylabel="relative brightness", ylim=(0, 1.05))
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "spot_fit.png", dpi=150)
    plt.close(fig)

    print(f"n (all)   = {n_all:.6f}  rmse = {rm_all:.5f}  ({all_a.size} px)")
    print(f"n (fwd)   = {n_fwd:.6f}  rmse = {rm_fwd:.5f}  ({fwd_a.size} px)")
    print(f"n (probe) = {n_probe:.6f}  rmse = {rm_probe:.5f}  ({probe_a.size} px)")
    print(f"written {out / 'calibration.json'} and {out / 'spot_fit.png'}")


if __name__ == "__main__":
    main()
