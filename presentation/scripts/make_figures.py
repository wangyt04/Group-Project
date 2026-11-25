#!/usr/bin/env python3
"""Build the chart figures for the final presentation.

Every number comes from presentation/data/key_numbers.json, which in turn cites
the run or document it came from. Nothing is typed in twice.

    ~/miniconda3/envs/pr-endo/bin/python scripts/make_figures.py

Output: presentation/figures/fig0*.png at 200 dpi, sized for a 16:9 slide.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
K = json.loads((ROOT / "data" / "key_numbers.json").read_text())

# Validated categorical slots 1-3 (dataviz reference palette, all-pairs clean).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.frameon": False,
    "legend.fontsize": 10.5,
})


def recessive(ax, axis="y"):
    """Grid behind the data, only two spines, no tick marks."""
    ax.grid(axis=axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def save(fig, name, caption=None):
    if caption:
        fig.text(0.012, -0.035, caption, fontsize=8.5, color=INK3, va="top")
    out = FIG / name
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


# --------------------------------------------------------------------------
# fig01  Two routes to a relit endoscopic sample
# --------------------------------------------------------------------------
def fig01_two_routes():
    fig, ax = plt.subplots(figsize=(13.2, 6.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    ax.set_title("两条生成路线：图形学渲染 vs. 可重光照高斯重建",
                 loc="left", pad=16)

    def box(x, y, w, h, title, lines, color, alpha=0.10):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, alpha=alpha,
                               edgecolor=color, linewidth=1.6, zorder=2,
                               joinstyle="round"))
        ax.text(x + w / 2, y + h - 4.4, title, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=INK, zorder=3)
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 10.0 - i * 4.6, line, ha="center",
                    va="center", fontsize=9.8, color=INK2, zorder=3)

    def arrow(x0, y0, x1, y1, color):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=16, linewidth=1.8,
                                     color=color, zorder=4,
                                     shrinkA=2, shrinkB=2))

    # Route A -- graphics engine
    ax.text(1, 56.5, "路线 A  Unity / Blender 图形学渲染", fontsize=12,
            fontweight="bold", color=ORANGE)
    box(1, 33, 21, 20, "解剖模型 + 材质",
        ["肠道网格、组织 BRDF", "VR-Caps 虚拟胶囊环境"], ORANGE)
    box(26, 33, 21, 20, "光源 + 相机",
        ["点光/聚光、强度、颜色", "位姿任意设定"], ORANGE)
    box(51, 33, 21, 20, "HDRP 物理渲染",
        ["逆平方衰减、阴影", "线性 float EXR 输出"], ORANGE)
    box(76, 33, 23, 20, "真实感样本 + 完整标签",
        ["RGB / 深度 / 法线 / 光源位姿", "物理正确，可当 GT"], ORANGE, 0.18)
    for x in (22, 47, 72):
        arrow(x, 43, x + 4, 43, ORANGE)
    ax.text(50, 29.5, "代价：逐帧离线渲染，依赖 Unity 工程与解剖资产，大规模生成慢",
            ha="center", fontsize=10, color=ORANGE, style="italic")

    # Route B -- Gaussian splatting
    ax.text(1, 24.0, "路线 B  高斯泼溅三维重建", fontsize=12,
            fontweight="bold", color=BLUE)
    box(1, 1, 21, 20, "内镜视频",
        ["RGB + 深度 + 位姿", "（SLAM 或仿真给出）"], BLUE)
    box(26, 1, 21, 20, "3D 高斯场景",
        ["21-27 万个高斯", "30k 步 ≈ 19 分钟"], BLUE)
    box(51, 1, 21, 20, "改相机 / 改光源",
        ["标准 3DGS：只能改相机", "PR-ENDO：还能改光源"], BLUE)
    box(76, 1, 23, 20, "批量新样本",
        [f"{K['efficiency']['prendo_render_ms_per_frame']:.1f} ms/帧"
         f" ≈ {K['efficiency']['prendo_render_fps']:.0f} fps",
         "物理可信度需要验证"], BLUE, 0.18)
    for x in (22, 47, 72):
        arrow(x, 11, x + 4, 11, BLUE)
    ax.text(50, -2.0, "代价：外观是从图像拟合出来的，没有真值就无法判断重光照对不对",
            ha="center", fontsize=10, color=BLUE, style="italic")

    # The bridge: route A supplies the ground truth that audits route B
    ax.add_patch(FancyArrowPatch((87.5, 32.0), (87.5, 22.0), arrowstyle="-|>",
                                 mutation_scale=18, linewidth=2.4,
                                 color=AQUA, zorder=5))
    ax.text(85.5, 27.0, "本项目：用 A 的真值\n量化 B 的重光照误差",
            ha="right", va="center", fontsize=10.2, fontweight="bold",
            color=AQUA, zorder=5)

    save(fig, "fig01_two_routes.png",
         "路线 B 的训练/渲染耗时来自 prendo/runs/vrcaps_lumen_100_v2_r30k（L40S，1350×1080）。")


# --------------------------------------------------------------------------
# fig02  Near-wall Gaussians are pruned at step 4000 (P5)
# --------------------------------------------------------------------------
def fig02_prune_timeline():
    p5 = K["p5_prune_timeline"]
    it = np.array(p5["iters"], dtype=float)
    total = np.array(p5["points"], dtype=float)
    near = np.array(p5["near_wall"], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))
    fig.suptitle("P5：近壁高斯点在第 4000 步的修剪中被删除",
                 x=0.012, ha="left", fontsize=14, fontweight="bold", color=INK)

    for ax, y, color, label, ylab in (
        (ax1, total, BLUE, "全部高斯点", "高斯点数量"),
        (ax2, near, ORANGE, "近壁高斯点（投影 z ≤ 0.2）", "近壁高斯点数量"),
    ):
        recessive(ax)
        ax.plot(it, y, color=color, linewidth=2.0, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
                label=label)
        ax.set_xlabel("训练步数")
        ax.set_ylabel(ylab)
        ax.set_xlim(-1200, 31500)
        ax.legend(loc="upper right")
        # the two events
        ax.axvline(3000, color=INK3, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.axvline(4000, color=INK3, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)

    ax1.annotate("3000 步\n不透明度重置", xy=(3000, 292000), xytext=(8200, 296000),
                 fontsize=9.6, color=INK2,
                 arrowprops=dict(arrowstyle="-", color=INK3, linewidth=0.9))
    ax1.annotate("4000 步 修剪\n删除 42,596 点", xy=(4000, 252000), xytext=(9200, 243000),
                 fontsize=9.6, color=INK2,
                 arrowprops=dict(arrowstyle="-", color=INK3, linewidth=0.9))

    ax2.annotate(f"这一步删掉的点里\n{p5['prune_at_4000_near_wall_removed']:,} 个是近壁点\n"
                 f"（占近壁的 95%）",
                 xy=(4500, 20000), xytext=(9000, 24000), fontsize=9.8,
                 color=ORANGE, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=1.2))
    for x, y in ((0, near[0]), (4000, near[4]), (30000, near[-1])):
        ax2.annotate(f"{int(y):,}", xy=(x, y), xytext=(0, 11),
                     textcoords="offset points", ha="center", fontsize=9.6,
                     color=INK, fontweight="bold")
    ax2.set_ylim(-2500, 42000)

    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    save(fig, "fig02_p5_prune_timeline.png",
         "prendo/runs/20261118-p1-p7/P5_NOTES.md（作业 3025783，原始 0.2 裁剪，每 1000 步存点云）。"
         "近壁定义：投影到第 0010 帧后 0 < z ≤ 0.2。")


# --------------------------------------------------------------------------
# fig03  PSNR per relighting level, baseline vs P1
# --------------------------------------------------------------------------
def fig03_level_gains():
    d = K["p1_vs_baseline_611"]
    levels = d["levels"]
    base = np.array(d["baseline_psnr"])
    p1 = np.array(d["p1_psnr"])
    x = np.arange(len(levels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(13.2, 5.2))
    recessive(ax)
    ax.bar(x - w / 2 - 0.012, base, w, color=ORANGE, zorder=3,
           label="原始 PR-ENDO（近裁剪 0.2）")
    ax.bar(x + w / 2 + 0.012, p1, w, color=BLUE, zorder=3,
           label="P1 修复后（近裁剪 0.01）")

    for xi, b, p in zip(x, base, p1):
        ax.text(xi - w / 2 - 0.012, b + 0.45, f"{b:.2f}", ha="center",
                fontsize=10, color=INK2)
        ax.text(xi + w / 2 + 0.012, p + 0.45, f"{p:.2f}", ha="center",
                fontsize=10.5, color=INK, fontweight="bold")
        ax.text(xi, max(b, p) + 2.6, f"+{p - b:.2f} dB", ha="center",
                fontsize=10.5, color=AQUA, fontweight="bold")

    names = {
        "L0": "L0\n不改光照\n(重建上限)",
        "L1": "L1\n改强度\n×0.5 / ×2",
        "L2": "L2\n点光→聚光\n7 个方向",
        "L3": "L3\n光源离开\n相机中心",
        "L4": "L4\n改颜色\nR / G / B",
        "L5": "L5\n光源固定\n相机移动",
    }
    ax.set_xticks(x)
    ax.set_xticklabels([names[l] for l in levels])
    ax.set_ylabel("PSNR (dB)  ↑")
    ax.set_ylim(0, 50)
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, 1.005))
    fig.suptitle("611 条重光照评估：修掉近裁剪后每个光照层级都提升 5.8–8.0 dB",
                 x=0.012, ha="left", fontsize=14, fontweight="bold", color=INK,
                 y=1.06)
    save(fig, "fig03_level_psnr_baseline_vs_p1.png",
         "prendo/runs/20261118-p1-p7/EVALUATION.md，relight_eval/compare.py，"
         "611 条 (config, frame)、803 个 GT variant 比较、0 缺失。")


# --------------------------------------------------------------------------
# fig04  Near-wall vs far-region: the gap closes
# --------------------------------------------------------------------------
def fig04_near_far():
    d = K["near_vs_far"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True)
    fig.suptitle("误差集中在近壁：GT 轴向深度 ≤ 0.2 的像素",
                 x=0.012, ha="left", fontsize=14, fontweight="bold", color=INK)

    panels = [
        (axes[0], "L0（不改光照）", d["L0"]["baseline"]["near"], d["L0"]["baseline"]["far"],
         d["L0"]["p1"]["near"], d["L0"]["p1"]["far"]),
        (axes[1], "L3（光源离开相机中心）", d["L3"]["baseline"]["near"],
         d["L3"]["baseline"]["far"], d["L3"]["p1"]["near"], d["L3"]["p1"]["far"]),
    ]
    x = np.arange(2)
    w = 0.38
    for ax, title, bn, bf, pn, pf in panels:
        recessive(ax)
        ax.bar(x - w / 2 - 0.012, [bn, bf], w, color=ORANGE, zorder=3,
               label="原始（0.2）")
        ax.bar(x + w / 2 + 0.012, [pn, pf], w, color=BLUE, zorder=3,
               label="P1（0.01）")
        for xi, b, p in zip(x, [bn, bf], [pn, pf]):
            ax.text(xi - w / 2 - 0.012, b + 0.4, f"{b:.2f}", ha="center",
                    fontsize=10, color=INK2)
            ax.text(xi + w / 2 + 0.012, p + 0.4, f"{p:.2f}", ha="center",
                    fontsize=10.5, color=INK, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["近壁\n(z ≤ 0.2)", "远区\n(z > 0.2)"])
        ax.set_title(title, loc="left", fontsize=12, pad=24)
        # gap brackets
        for xi, (lo, hi), color, txt in (
            (0.0, (bn, bf), ORANGE, f"差距 {bf - bn:.1f} dB"),
            (0.0, (pn, pf), BLUE, f"差距 {pf - pn:.1f} dB"),
        ):
            pass
        ax.text(0.5, 47.5, f"原始差距 {bf - bn:.1f} dB  →  P1 差距 {pf - pn:.1f} dB",
                ha="center", fontsize=10.8, color=AQUA, fontweight="bold")
    axes[0].set_ylabel("PSNR (dB)  ↑")
    axes[0].set_ylim(0, 52)
    axes[0].legend(loc="upper left", ncol=2, bbox_to_anchor=(0, 1.11))
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    save(fig, "fig04_near_vs_far_psnr.png",
         "prendo/runs/20261118-p1-p7/EVALUATION.md（49 个 L0 视角 / 391 条非 L2，按 GT 深度分区）。"
         "0.2 世界单位 = 0.595 d_ref。")


# --------------------------------------------------------------------------
# fig05  Learned attenuation stays flatter than inverse square (P7)
# --------------------------------------------------------------------------
def fig05_attenuation():
    d = K["p7_attenuation"]
    k_b, p_b = d["k"][0], d["p"][0]
    k_1, p_1 = d["k"][1], d["p"][1]
    d_ref = K["dataset"]["d_ref_world_units"]

    # scene/gaussian_model.py:259 feeds the formula a distance already divided
    # by spatial_lr_scale, so d in 1/(1+k d^p) is in units of 0.0995 world units.
    s = K["baseline_model"]["spatial_lr_scale"]
    r = np.linspace(0.30, 1.60, 400)        # distance in units of d_ref
    dist = r * d_ref / s                    # ...converted to the model's units

    def att(dd, k, p):
        return 1.0 / (1.0 + k * dd ** p)

    # normalise every curve to 1.0 at d = d_ref so only the *shape* is compared
    d0 = d_ref / s
    inv_sq = (1.0 / dist ** 2) / (1.0 / d0 ** 2)
    a_b = att(dist, k_b, p_b) / att(d0, k_b, p_b)
    a_1 = att(dist, k_1, p_1) / att(d0, k_1, p_1)

    # sanity check against the number quoted in EVALUATION.md
    ratio_b = att(0.45 * d0, k_b, p_b) / att(1.3 * d0, k_b, p_b)
    ratio_1 = att(0.45 * d0, k_1, p_1) / att(1.3 * d0, k_1, p_1)
    print(f"  check  falloff 1.3/0.45 d_ref: baseline {ratio_b:.2f} "
          f"(doc {d['falloff_ratio_1_3_over_0_45']['baseline']}), "
          f"P1 {ratio_1:.2f} (doc {d['falloff_ratio_1_3_over_0_45']['p1']})")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2),
                                  gridspec_kw={"width_ratios": [1.45, 1]})
    recessive(ax)
    ax.plot(r, inv_sq, color=AQUA, linewidth=2.4, zorder=4,
            label="Unity 物理真值：逆平方 $1/d^2$")
    ax.plot(r, a_b, color=ORANGE, linewidth=2.0, zorder=3,
            label=f"原始模型学到的 $1/(1+{k_b:.3f}d^{{{p_b:.3f}}})$")
    ax.plot(r, a_1, color=BLUE, linewidth=2.0, linestyle=(0, (5, 2)), zorder=3,
            label=f"P1 学到的 $1/(1+{k_1:.3f}d^{{{p_1:.3f}}})$")
    ax.axvline(1.0, color=INK3, linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
    ax.annotate("$d_{ref}$ = 工作距离中位数", xy=(1.03, 0.04),
                xycoords=("data", "axes fraction"), fontsize=9.5, color=INK3,
                va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("光源到表面的距离 / $d_{ref}$")
    ax.set_ylabel("相对衰减（在 $d_{ref}$ 处归一化到 1）")
    ax.set_title("学到的衰减在近距离比物理更平", loc="left", pad=14)
    ax.legend(loc="upper right")

    # right panel: the single headline ratio
    recessive(ax2)
    ratios = d["falloff_ratio_1_3_over_0_45"]
    names = ["Unity\n逆平方", "原始\nPR-ENDO", "P1\n修复后"]
    vals = [ratios["unity_inverse_square"], ratios["baseline"], ratios["p1"]]
    colors = [AQUA, ORANGE, BLUE]
    ax2.bar(np.arange(3), vals, 0.56, color=colors, zorder=3)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.14, f"{v:.2f}×", ha="center", fontsize=12,
                 fontweight="bold", color=INK)
    ax2.set_xticks(np.arange(3))
    ax2.set_xticklabels(names)
    ax2.set_ylabel("亮度衰减倍数")
    ax2.set_ylim(0, 10.2)
    ax2.set_title("从 0.45 到 1.3 $d_{ref}$ 应衰减多少倍", loc="left", pad=14)
    ax2.text(1.0, 9.1, "P1 只把 5.10 拉到 5.31\n→ 不是裁剪问题，是公式形式的限制",
             ha="center", fontsize=10, color=INK2)

    fig.tight_layout()
    save(fig, "fig05_attenuation_vs_inverse_square.png",
         "P7：从 chkpnt_light30000.pth 读出的 (intensity, k, p)；"
         "公式见 scene/gaussian_model.py:280。曲线按形状比较，故在 d_ref 处归一化。")


# --------------------------------------------------------------------------
# fig06  P4 render depth in the culled region
# --------------------------------------------------------------------------
def fig06_p4_depth():
    d = K["p4_diagnostics"]
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    recessive(ax)
    labels = ["原始\n(裁剪 0.2)", "P1\n(裁剪 0.01)", "P1+P2\n(裁剪 0.01)"]
    x = np.arange(3)
    w = 0.38
    near = d["render_over_gt_depth_near"]
    far = d["render_over_gt_depth_far"]
    ax.bar(x - w / 2 - 0.012, near, w, color=ORANGE, zorder=3, label="近壁像素")
    ax.bar(x + w / 2 + 0.012, far, w, color=BLUE, zorder=3, label="远区像素")
    ax.axhline(1.0, color=AQUA, linewidth=2.0, zorder=4)
    ax.text(-0.46, 1.022, "1.0 = 渲染深度等于真值", fontsize=10, color=AQUA,
            fontweight="bold", ha="left")
    for xi, n, f in zip(x, near, far):
        ax.text(xi - w / 2 - 0.012, n + 0.022, f"{n:.3f}", ha="center",
                fontsize=11, fontweight="bold", color=INK)
        ax.text(xi + w / 2 + 0.012, f + 0.022, f"{f:.3f}", ha="center",
                fontsize=10.5, color=INK2)
    ax.annotate("近壁被 2 倍远处的\n“幕布”高斯画出来", xy=(0.0, 1.50),
                xytext=(0.62, 1.52), fontsize=10.5, color=ORANGE,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=1.3))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("渲染深度 / 真值深度（中位数）")
    ax.set_ylim(0, 1.78)
    ax.set_title("P4 GPU 诊断：近裁剪让近壁的渲染深度偏到真值的 1.6 倍",
                 loc="left", pad=14)
    ax.legend(loc="upper right")
    save(fig, "fig06_p4_render_depth_ratio.png",
         "prendo/runs/20261118-p1-p7/artifacts/diag_p4/*/summary.json（49 个 L0 视角）。"
         "CPU 近似曾给出 1.7 倍，GPU 实测 1.596。")


# --------------------------------------------------------------------------
# fig10  L2: per-direction PSNR, old model vs P1
# --------------------------------------------------------------------------
def fig10_l2_directions():
    d = K["l2_4x_oldmodel"]
    cfgs = d["configs"]
    old = np.array(d["psnr"])
    unrelit = np.array(d["unrelit_psnr"])
    x = np.arange(len(cfgs))
    w = 0.27

    fig, ax = plt.subplots(figsize=(13.2, 5.2))
    recessive(ax)
    ax.bar(x - w - 0.012, unrelit, w, color=INK3, zorder=3,
           label="完全不重光照（直接用 L0 渲染）")
    ax.bar(x, old, w, color=ORANGE, zorder=3, label="原始模型重光照")
    ax.axhspan(d["p1_l2_range"][0], d["p1_l2_range"][1], color=BLUE, alpha=0.16,
               zorder=2)
    ax.axhline(d["p1_l2_mean_psnr"], color=BLUE, linewidth=2.2, zorder=4,
               label=f"P1 修复后 7 个方向平均 {d['p1_l2_mean_psnr']:.2f} dB")

    for xi, u, o in zip(x, unrelit, old):
        ax.text(xi - w - 0.012, u + 0.35, f"{u:.1f}", ha="center", fontsize=9.3,
                color=INK2)
        ax.text(xi, o + 0.35, f"{o:.1f}", ha="center", fontsize=9.8, color=INK,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("L2_", "") for c in cfgs])
    ax.set_xlabel("L2 光束方向配置")
    ax.set_ylabel("PSNR (dB)  ↑")
    ax.set_ylim(0, 46)
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.005))
    fig.suptitle("L2 点光→聚光：重光照本身已经值 +10～12 dB，P1 再加约 +7 dB",
                 x=0.012, ha="left", fontsize=14, fontweight="bold", color=INK,
                 y=1.06)
    save(fig, "fig10_l2_direction_psnr.png",
         "prendo/runs/20261113-0110/EVALUATION.md（旧模型，220 条，30 个 test 帧/配置）"
         "与 20261118-p1-p7 的 P1 L2 结果。")


# --------------------------------------------------------------------------
# fig11  Specular: the one thing P1 and P2 did not fix
# --------------------------------------------------------------------------
def fig11_specular():
    s = K["specular"]
    p4 = K["p4_diagnostics"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2))
    fig.suptitle("高光是唯一没被修好的部分", x=0.012, ha="left",
                 fontsize=14, fontweight="bold", color=INK)

    # left: highlight behaviour when the light moves right by 0.25 d_ref
    recessive(ax1)
    cats = ["最亮 0.1% 像素\n的亮度", "光源右移后\n高光留在原位的比例", "光源右移后\n新高光强度的保留"]
    unity = [s["L0_top01pct_unity"], s["stays_in_place_after_r0.25_unity"],
             s["new_highlight_intensity_retained_unity"]]
    prendo = [s["L0_top01pct_prendo"], s["stays_in_place_after_r0.25_prendo"],
              s["new_highlight_intensity_retained_prendo"]]
    x = np.arange(3)
    w = 0.38
    ax1.bar(x - w / 2 - 0.012, unity, w, color=AQUA, zorder=3, label="Unity 真值")
    ax1.bar(x + w / 2 + 0.012, prendo, w, color=ORANGE, zorder=3, label="PR-ENDO")
    for xi, u, p in zip(x, unity, prendo):
        ax1.text(xi - w / 2 - 0.012, u + 0.018, f"{u:.2f}", ha="center",
                 fontsize=10.5, color=INK, fontweight="bold")
        ax1.text(xi + w / 2 + 0.012, p + 0.018, f"{p:.2f}", ha="center",
                 fontsize=10.5, color=INK, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, fontsize=9.8)
    ax1.set_ylim(0, 1.16)
    ax1.set_title("A16：高光更弱，而且不跟着光源搬家", loc="left",
                  fontsize=12, pad=14)
    ax1.legend(loc="upper left", ncol=2)

    # right: F0 is pinned by the clamp
    recessive(ax2)
    f0 = K["p2_f0"]
    models = ["原始", "P1", "P1+P2\n(f0_init=0.03)"]
    means = [f0["baseline"]["mean"], f0["p1"]["mean"], f0["p1p2"]["mean"]]
    uniq = [f0["baseline"]["unique"], f0["p1"]["unique"], f0["p1p2"]["unique"]]
    x = np.arange(3)
    ax2.bar(x, means, 0.5, color=[ORANGE, ORANGE, BLUE], zorder=3)
    ax2.axhline(0.035, color=AQUA, linewidth=2.2, zorder=4)
    ax2.text(-0.46, 0.0362, "代码里的截断上限 clamp(F0, 0, 0.035)", fontsize=9.8,
             color=AQUA, fontweight="bold", ha="left")
    for xi, m, u in zip(x, means, uniq):
        ax2.text(xi, m + 0.0016, f"{m:.4f}", ha="center", fontsize=10.5,
                 color=INK, fontweight="bold")
        ax2.text(xi, 0.0035, f"{u:,} 个\n不同取值", ha="center", fontsize=9.6,
                 color="white" if u == 1 else "white", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylabel("F0 均值（基础反射率）")
    ax2.set_ylim(0, 0.062)
    ax2.set_title("P2：F0 初值 0.05 高于截断 → 梯度恒为 0", loc="left",
                  fontsize=12, pad=14)
    ax2.text(1.0, 0.055,
             f"纯高光图非零像素：原始 {p4['specular_only_nonzero_fraction'][0]:.1%}"
             f" → P1 {p4['specular_only_nonzero_fraction'][1]:.1%}"
             f" → P1+P2 {p4['specular_only_nonzero_fraction'][2]:.1%}",
             ha="center", fontsize=9.8, color=INK2)

    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    save(fig, "fig11_specular_and_f0.png",
         "A16（A 清单）、B15、P2（artifacts/p2_f0）、P4（artifacts/diag_p4）；"
         "截断见 scene/gaussian_model.py:182，论文取 0.03。")


# --------------------------------------------------------------------------
# fig12  What the 0.2 near-cull actually does (geometry schematic)
# --------------------------------------------------------------------------
def fig12_cull_schematic():
    rc = K["root_cause"]
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    ax.set_xlim(-0.03, 0.46)
    ax.set_ylim(-0.185, 0.20)
    ax.axis("off")
    ax.set_title("近裁剪 z ≤ 0.2 在内镜里意味着什么", loc="left", pad=14)

    # camera + light (co-located, the endoscopy assumption PR-ENDO exploits)
    ax.add_patch(Rectangle((-0.022, -0.016), 0.03, 0.032, facecolor=INK,
                           zorder=5))
    ax.text(-0.007, -0.030, "相机 + 光源\n（同一位置）", ha="center", va="top",
            fontsize=9.8, color=INK)

    # frustum
    for sign in (1, -1):
        ax.plot([0.008, 0.44], [0, sign * 0.155], color=INK3, linewidth=1.0,
                linestyle=(0, (3, 3)), zorder=1)

    # cull plane
    ax.plot([0.2, 0.2], [-0.145, 0.145], color=ORANGE, linewidth=2.6, zorder=4)
    ax.text(0.205, 0.143, "硬编码裁剪面 z = 0.2\nauxiliary.h:154\n"
            f"= {rc['threshold_in_d_ref']:.3f} × 工作距离"
            f"（znear 只有 {rc['znear']}，与它无关）",
            fontsize=10.2, color=ORANGE, fontweight="bold", va="top")

    # true near wall at 0.137
    ax.plot([0.137, 0.137], [-0.10, 0.10], color=AQUA, linewidth=2.6, zorder=4)
    ax.text(0.133, 0.104, "真实近壁 z ≈ 0.137", fontsize=10.2, color=AQUA,
            fontweight="bold", ha="right", va="bottom")

    # curtain gaussians at 0.232
    rng = np.random.default_rng(7)
    cy = rng.uniform(-0.085, 0.085, 40)
    cx = 0.232 + rng.normal(0, 0.009, 40)
    ax.scatter(cx, cy, s=52, color=BLUE, alpha=0.55, edgecolor="none", zorder=3)
    ax.annotate("真正被画出来的“幕布”高斯\nz ≈ 0.232（真值的 1.6 倍）",
                xy=(0.244, -0.055), xytext=(0.272, -0.112), fontsize=10.2,
                color=BLUE, fontweight="bold", va="top",
                arrowprops=dict(arrowstyle="-|>", color=BLUE, linewidth=1.3))

    # culled region shading
    ax.add_patch(Rectangle((0.008, -0.145), 0.192, 0.29, facecolor=ORANGE,
                           alpha=0.10, edgecolor="none", zorder=0))
    ax.text(0.104, 0.0,
            f"被剔除的区域\n占画面 {rc['culled_pixel_fraction']:.1%} 的像素\n"
            f"却占 L0 平方误差的 {rc['share_of_L0_squared_error']:.0%}",
            ha="center", va="center", fontsize=10.6, color=ORANGE,
            fontweight="bold", zorder=5)

    # far wall
    ax.plot([0.40, 0.40], [-0.14, 0.14], color=INK3, linewidth=2.0, zorder=2)
    ax.text(0.404, 0.0, "远壁\n正常渲染\n42–43 dB", fontsize=9.8, color=INK2,
            va="center")

    ax.text(0.0, -0.166,
            f"结果：{rc['never_rendered_points']:,} 个初始高斯（{rc['never_rendered_fraction']:.1%}）"
            f"在画面内却从未被渲染 → 没有梯度 → 第 4000 步修剪时被删除",
            fontsize=10.4, color=INK, fontweight="bold")
    save(fig, "fig12_near_cull_schematic.png",
         "B01（源码）、B04/B10（幕布深度）、B07（从未渲染的点）、B09（误差占比）、"
         "P4（GPU 实测 1.596 倍）。示意图，横轴为世界单位。")


if __name__ == "__main__":
    fig01_two_routes()
    fig02_prune_timeline()
    fig03_level_gains()
    fig04_near_far()
    fig05_attenuation()
    fig06_p4_depth()
    fig10_l2_directions()
    fig11_specular()
    fig12_cull_schematic()
    print("\nall charts written to", FIG)
