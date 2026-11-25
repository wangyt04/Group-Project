#!/usr/bin/env python3
"""Build the stacked image comparisons for the slides.

Three rows per column, always in the same order:

    row 1  原始 PR-ENDO   baseline model, rasteriser near-cull 0.2
    row 2  P1 修复后       same data/hyperparameters, near-cull 0.01
    row 3  Unity 真值      VR-Caps HDRP ground truth

Per-tile PSNR is computed the same way compare.py does it: GT is linear EXR,
encoded with E(x) = clamp(x,0,1)^(1/2.2); predictions are already display-space
PNGs; valid pixels are depth > 0.

    ~/miniconda3/envs/pr-endo/bin/python scripts/make_image_panels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]                       # ~/Documents/my_repo
sys.path.insert(0, str(REPO / "Group-Project" / "relight_eval"))
import relight_io as io                      # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

EVAL_ROOT = REPO / "PR-ENDO-data" / "P1_611_eval" / "VRCaps_lumen_100_v2"
BASE_NONL2 = REPO / "PR-ENDO" / "artifacts" / "pred_VRCaps_lumen_100_v2_20261112-0332"
BASE_L2 = REPO / "PR-ENDO" / "artifacts" / "pred_VRCaps_lumen_100_v2_20261113-0110"
P1_NONL2 = REPO / "PR-ENDO-data" / "P1_download" / "nonl2"
P1_L2 = REPO / "PR-ENDO-data" / "P1_download" / "l2"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

ds = io.Dataset(EVAL_ROOT)


def is_l2(cfg):
    return cfg.startswith("L2_")


def load_gt(cfg, frame):
    variants = ds.variants(cfg)
    variant = "noshadow" if "noshadow" in variants else variants[0]
    return io.encode(ds.load_gt(cfg, variant, frame))


def load_pred(root, cfg, frame):
    p = Path(root) / cfg / f"{frame}.png"
    return io.read_png(p) if p.exists() else None


def psnr(pred, gt, valid):
    if pred is None or not valid.any():
        return float("nan")
    err = ((pred - gt) ** 2)[valid].mean()
    return float(-10 * np.log10(max(float(err), 1e-12)))


def crop_for_display(img, scale=3):
    """Downsample 1350x1080 to something a slide can hold."""
    return img[::scale, ::scale]


def panel(items, outname, suptitle, caption, near_overlay=False):
    """items: list of (config, frame, column caption)."""
    n = len(items)
    rows = ["原始 PR-ENDO\n(近裁剪 0.2)", "P1 修复后\n(近裁剪 0.01)", "Unity 真值\n(VR-Caps HDRP)"]
    fig, axes = plt.subplots(3, n, figsize=(2.95 * n + 1.5, 8.2),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.03,
                                          "left": 0.085, "right": 0.995,
                                          "top": 0.885, "bottom": 0.055})
    if n == 1:
        axes = axes.reshape(3, 1)

    for j, (cfg, frame, col_caption) in enumerate(items):
        gt = load_gt(cfg, frame)
        depth = ds.load_depth(frame)
        valid = np.isfinite(depth) & (depth > 0)
        base_root = BASE_L2 if is_l2(cfg) else BASE_NONL2
        p1_root = P1_L2 if is_l2(cfg) else P1_NONL2
        base = load_pred(base_root, cfg, frame)
        p1 = load_pred(p1_root, cfg, frame)

        near = valid & (depth <= 0.2)
        tiles = [
            (base, f"{psnr(base, gt, valid):.1f} dB", ORANGE),
            (p1, f"{psnr(p1, gt, valid):.1f} dB", BLUE),
            (gt, "参考", AQUA),
        ]
        for i, (img, tag, color) in enumerate(tiles):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(color); s.set_linewidth(2.0)
            if img is None:
                ax.text(0.5, 0.5, "缺失", ha="center", va="center",
                        transform=ax.transAxes, color=INK3)
                continue
            ax.imshow(np.clip(crop_for_display(img), 0, 1))
            if near_overlay:
                m = crop_for_display(near).astype(float)
                ax.contour(m, levels=[0.5], colors=["#ffd166"], linewidths=1.3)
            ax.text(0.015, 0.975, tag, transform=ax.transAxes, va="top",
                    fontsize=12, fontweight="bold", color="white",
                    bbox=dict(facecolor=color, edgecolor="none",
                              boxstyle="round,pad=0.26"))
            if i == 0:
                ax.set_title(col_caption, fontsize=11.5, color=INK,
                             fontweight="bold", pad=7)
        if j == 0:
            for i in range(3):
                axes[i, 0].set_ylabel(rows[i], fontsize=11.5, color=INK,
                                      fontweight="bold", labelpad=10)

    fig.suptitle(suptitle, x=0.085, ha="left", fontsize=16, fontweight="bold",
                 color=INK, y=0.965)
    fig.text(0.085, 0.918,
             "每格左上角是该帧相对 Unity 真值的 PSNR（显示空间，有效像素 depth > 0）",
             fontsize=10, color=INK2)
    fig.text(0.012, 0.012, caption, fontsize=8.5, color=INK3)
    out = FIG / outname
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


# --------------------------------------------------------------------------
# Depth-band curve: where exactly does the error live?
# --------------------------------------------------------------------------
def depth_band_figure(frames, outname):
    """L0 PSNR as a function of GT axial depth, baseline vs P1.

    This is the measurement that puts the jump exactly at the 0.2 cull plane.
    """
    edges = np.array([0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25,
                      0.30, 0.35, 0.45, 0.60])
    se_b = np.zeros(len(edges) - 1)
    se_p = np.zeros(len(edges) - 1)
    cnt = np.zeros(len(edges) - 1)

    for frame in frames:
        gt = load_gt("L0", frame)
        depth = ds.load_depth(frame)
        valid = np.isfinite(depth) & (depth > 0)
        base = load_pred(BASE_NONL2, "L0", frame)
        p1 = load_pred(P1_NONL2, "L0", frame)
        if base is None or p1 is None:
            continue
        eb = ((base - gt) ** 2).mean(axis=2)
        ep = ((p1 - gt) ** 2).mean(axis=2)
        idx = np.digitize(depth, edges) - 1
        for b in range(len(edges) - 1):
            m = valid & (idx == b)
            if not m.any():
                continue
            se_b[b] += eb[m].sum(); se_p[b] += ep[m].sum(); cnt[b] += m.sum()

    ok = cnt > 0
    centres = 0.5 * (edges[:-1] + edges[1:])
    pb = -10 * np.log10(np.maximum(se_b[ok] / cnt[ok], 1e-12))
    pp = -10 * np.log10(np.maximum(se_p[ok] / cnt[ok], 1e-12))
    xs = centres[ok]

    fig, ax = plt.subplots(figsize=(12.0, 5.6))
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#e4e3df"); ax.spines["bottom"].set_color("#e4e3df")
    ax.tick_params(length=0, colors=INK2)

    ax.axvspan(edges[0], 0.2, color=ORANGE, alpha=0.09, zorder=1)
    ax.axvline(0.2, color=ORANGE, linewidth=2.4, zorder=2)
    ax.text(0.197, 20.0, "裁剪面 z = 0.2", rotation=90, ha="right", va="bottom",
            fontsize=11, color=ORANGE, fontweight="bold")
    ax.text(0.137, 19.2, "← 被剔除的近壁", fontsize=11, color=ORANGE,
            fontweight="bold", ha="center")

    ax.plot(xs, pb, color=ORANGE, linewidth=2.2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=4,
            label="原始 PR-ENDO（近裁剪 0.2）")
    ax.plot(xs, pp, color=BLUE, linewidth=2.2, marker="s", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=4,
            label="P1 修复后（近裁剪 0.01）")

    ax.set_xlabel("Unity 真值的轴向深度（世界单位）")
    ax.set_ylabel("该深度区间内的 L0 PSNR (dB)  ↑")
    ax.set_title(f"L0 误差的跃变点正好落在 0.2 的裁剪面上（{len(frames)} 个 L0 视角逐像素统计）",
                 loc="left", fontsize=14, fontweight="bold", color=INK, pad=14)
    ax.legend(loc="lower right")
    ax.set_ylim(18, 55)
    fig.text(0.012, 0.012,
             "本图由 scripts/make_image_panels.py 直接从真值 EXR、基线预测与 P1 预测重新计算，"
             "不是从任何汇总表抄来的。",
             fontsize=8.5, color=INK3)
    out = FIG / outname
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")
    for x, b, p in zip(xs, pb, pp):
        print(f"    depth {x:.3f}  baseline {b:5.2f} dB   P1 {p:5.2f} dB")


# --------------------------------------------------------------------------
# Error maps: where the two models differ from the ground truth
# --------------------------------------------------------------------------
def error_maps(items, outname, suptitle, caption, vmax=0.25):
    """items: list of (config, frame, column caption). 3 rows: GT, base err, P1 err."""
    n = len(items)
    fig, axes = plt.subplots(3, n, figsize=(2.95 * n + 1.6, 8.4),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.03,
                                          "left": 0.09, "right": 0.93,
                                          "top": 0.885, "bottom": 0.055})
    if n == 1:
        axes = axes.reshape(3, 1)
    rows = ["Unity 真值", "原始 PR-ENDO\n的误差 |预测 − 真值|", "P1 修复后\n的误差 |预测 − 真值|"]
    im = None
    for j, (cfg, frame, col_caption) in enumerate(items):
        gt = load_gt(cfg, frame)
        depth = ds.load_depth(frame)
        valid = np.isfinite(depth) & (depth > 0)
        base = load_pred(BASE_L2 if is_l2(cfg) else BASE_NONL2, cfg, frame)
        p1 = load_pred(P1_L2 if is_l2(cfg) else P1_NONL2, cfg, frame)

        axes[0, j].imshow(np.clip(crop_for_display(gt), 0, 1))
        axes[0, j].set_title(col_caption, fontsize=11.5, color=INK,
                             fontweight="bold", pad=7)
        for i, (pred, color) in enumerate(((base, ORANGE), (p1, BLUE)), start=1):
            ax = axes[i, j]
            if pred is None:
                ax.text(0.5, 0.5, "缺失", ha="center", va="center",
                        transform=ax.transAxes, color=INK3)
                continue
            e = np.abs(pred - gt).mean(axis=2)
            e = np.where(valid, e, np.nan)
            im = ax.imshow(crop_for_display(e), cmap="magma", vmin=0, vmax=vmax)
            sq = np.where(valid, e, 0.0) ** 2
            total = float(sq.sum())
            share = float(sq[valid & (depth <= 0.2)].sum())
            ax.text(0.015, 0.975,
                    f"近壁占平方误差 {share / max(total, 1e-12):.0%}",
                    transform=ax.transAxes, va="top", fontsize=11,
                    fontweight="bold", color="white",
                    bbox=dict(facecolor=color, edgecolor="none",
                              boxstyle="round,pad=0.26"))
        for i in range(3):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for sp in axes[i, j].spines.values():
                sp.set_color("#d8d7d2")
        if j == 0:
            for i in range(3):
                axes[i, 0].set_ylabel(rows[i], fontsize=11.5, color=INK,
                                      fontweight="bold", labelpad=10)
    if im is not None:
        cax = fig.add_axes([0.945, 0.06, 0.014, 0.52])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("绝对误差（显示空间）", fontsize=10, color=INK2)
        cb.ax.tick_params(labelsize=9, colors=INK2, length=0)
        cb.outline.set_visible(False)
    fig.suptitle(suptitle, x=0.09, ha="left", fontsize=16, fontweight="bold",
                 color=INK, y=0.965)
    fig.text(0.09, 0.918, "越亮 = 误差越大；无效像素（depth ≤ 0）留白",
             fontsize=10, color=INK2)
    fig.text(0.012, 0.012, caption, fontsize=8.5, color=INK3)
    out = FIG / outname
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    l0_frames = sorted(p.stem for p in (P1_NONL2 / "L0").glob("*.png"))
    print(f"{len(l0_frames)} L0 frames available")

    panel([("L0", "0048", "第 0048 帧"),
           ("L0", "0089", "第 0089 帧"),
           ("L0", "0099", "第 0099 帧")],
          "fig13_L0_baseline_p1_gt.png",
          "L0（完全不改光照）：近裁剪让近壁糊成一片",
          "预测：artifacts/pred_..._20261112-0332（基线）与 PR-ENDO-data/P1_download（P1）；"
          "真值：P1_611_eval/.../relight/gt/L0/noshadow。黄色轮廓圈出的是 GT 深度 ≤ 0.2 的近壁区域，也就是被光栅化器剔除的那部分。",
          near_overlay=True)

    panel([("L2_yaw+20", "0048", "L2 yaw +20°"),
           ("L2_yaw-20", "0048", "L2 yaw −20°"),
           ("L2_pitch+20", "0048", "L2 pitch +20°")],
          "fig14_L2_directions_baseline_p1_gt.png",
          "L2（点光 → 4× 聚光，光束转向）：方向对，但近壁拖累了分数",
          "同一帧 0048 的三个光束方向。基线 L2 来自 run 20261113-0110（n = 2.4005），"
          "P1 L2 来自 20261118-p1-p7 的 220 条渲染。")

    panel([("L3_f+0.25", "0048", "L3 光源前移 0.25 d_ref"),
           ("L3_r0.50", "0048", "L3 光源右移 0.50 d_ref"),
           ("L4_red", "0048", "L4 红光")],
          "fig15_L3_L4_baseline_p1_gt.png",
          "L3 / L4：光源离开相机中心是最难的一类，改颜色则几乎无损",
          "L3 把光源移出相机光心，这正是论文承认的“光源离相机较远”场景；"
          "L4 只缩放光源 RGB，误差基本等于 L0 的重建误差。")

    error_maps([("L0", "0099", "L0 第 0099 帧"),
                ("L3_f+0.25", "0048", "L3 前移 第 0048 帧"),
                ("L2_yaw+20", "0048", "L2 yaw+20 第 0048 帧")],
               "fig17_error_maps.png",
               "误差图：原始模型的误差几乎全部堆在近壁，P1 把它清掉了",
               "误差 = 显示空间三通道绝对差的均值；标注的百分比是 GT 深度 ≤ 0.2 的像素"
               "在该帧全部平方误差中的占比。")

    depth_band_figure(l0_frames, "fig16_psnr_vs_depth_band.png")
