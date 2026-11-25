# presentation/ — 期末演示材料

内镜场景的重光照新样本生成与数据集构建。这个目录收齐了演示需要的全部内容：总结、幻灯片大纲、
图表、动图、数据表、GUI 演示方案，以及重新生成所有图表的脚本。

## 先读哪个

| 文件 | 内容 |
|---|---|
| **[`演示总结_中文版.md`](演示总结_中文版.md)** | **主文档。** 完整的中文内容总结：背景、两条路线、PR-ENDO 的创新与缺陷、我们的实验、结论 |
| [`幻灯片大纲.md`](幻灯片大纲.md) | 18 页的逐页大纲，标了每页用哪张图、哪些页可以砍，末尾附预期问题与答案 |
| [`demo/DEMO_GUIDE.md`](demo/DEMO_GUIDE.md) | 两个 GUI 怎么演示（结论：录屏，不现场跑），含录屏分镜和 ffmpeg 命令 |

## 目录结构

```
presentation/
├── 演示总结_中文版.md          ← 主文档
├── 幻灯片大纲.md               ← 逐页大纲 + Q&A 预案
├── figures/                    ← 17 张演示图（PNG，200 dpi，16:9）
├── gifs/                       ← 8 个 P1 预测 vs Unity 真值 的上下对比动图
├── demo/
│   ├── DEMO_GUIDE.md
│   └── clips/                  ← EndoGSLAM 的现成视频（穿行 / 环绕 / 真值对比）
├── tables/                     ← 评估输出的 CSV
├── data/key_numbers.json       ← 所有数字的唯一来源，每项都标了出处
└── scripts/                    ← 重新生成图表的脚本
```

## 图表清单

**三行图像对比的行序在所有图里都一样：上 = 原始 PR-ENDO，中 = P1 修复后，下 = Unity 真值。**

| 图 | 内容 | 出处 |
|---|---|---|
| `fig01_two_routes.png` | 两条生成路线的对比流程图 | 新建 |
| `fig02_p5_prune_timeline.png` | P5：近壁高斯点在第 4000 步被删 | `runs/20261118-p1-p7/P5_NOTES.md` |
| `fig03_level_psnr_baseline_vs_p1.png` | 6 个光照层级的 PSNR，原始 vs P1 | 611 条评估 |
| `fig04_near_vs_far_psnr.png` | 近壁 / 远区差距从 14.9 dB 收到 2.1 dB | 同上 |
| `fig05_attenuation_vs_inverse_square.png` | 学到的衰减 vs 物理逆平方 | P7，从 `chkpnt_light30000.pth` 读参数 |
| `fig06_p4_render_depth_ratio.png` | P4：近壁渲染深度 1.596 → 0.985 | `artifacts/diag_p4/*/summary.json` |
| `fig07_L2_direction_effect_maps.png` | GT 上的 L2/L0 归一化亮度比图 | `runs/20261113-0110` 评估产物 |
| `fig08_L2_compare_5x2_frame0048_oldmodel.png` | 旧模型 L2 五宫格 | 同上 |
| `fig09_spot_power_calibration_fit.png` | 聚光指数 n = 2.4005 的标定拟合 | 同上 |
| `fig10_l2_direction_psnr.png` | L2 各方向 PSNR，含"完全不重光照"基线 | 两次 run |
| `fig11_specular_and_f0.png` | 高光偏弱 + F0 被截断 | A16 / B15 / P2 / P4 |
| `fig12_near_cull_schematic.png` | 近裁剪在内镜几何里意味着什么 | B01/B04/B07/B09/B10 + P4 |
| `fig13_L0_baseline_p1_gt.png` | L0 三行对比，黄色轮廓圈出被剔除的近壁 | 原始图像重新渲染 |
| `fig14_L2_directions_baseline_p1_gt.png` | L2 三个光束方向的三行对比 | 同上 |
| `fig15_L3_L4_baseline_p1_gt.png` | L3 / L4 的三行对比 | 同上 |
| `fig16_psnr_vs_depth_band.png` | **误差跃变精确落在 0.2**（从原始图像独立复算） | 同上 |
| `fig17_error_maps.png` | 误差图：误差几乎全堆在近壁 | 同上 |

`fig13`–`fig17` 不是从任何汇总表抄来的，是脚本直接读真值 EXR、基线预测、P1 预测重新算的。
`fig05` 的脚本会把算出来的衰减比和 `EVALUATION.md` 里记录的值打印出来对照（5.10 / 5.31，吻合）。

## 重新生成图表

需要 `pr-endo` 这个 conda 环境（numpy / matplotlib / opencv / OpenEXR / PIL 都齐）：

```bash
cd ~/Documents/my_repo/Group-Project/presentation

# 图表（只依赖 data/key_numbers.json，几秒钟）
~/miniconda3/envs/pr-endo/bin/python scripts/make_figures.py

# 图像对比与深度分档（要读原始 PNG/EXR，约 10 秒）
~/miniconda3/envs/pr-endo/bin/python scripts/make_image_panels.py
```

`make_image_panels.py` 依赖这几个目录存在：

- `PR-ENDO-data/P1_611_eval/VRCaps_lumen_100_v2/` — 真值与深度
- `PR-ENDO/artifacts/pred_VRCaps_lumen_100_v2_20261112-0332/` — 基线非 L2 预测
- `PR-ENDO/artifacts/pred_VRCaps_lumen_100_v2_20261113-0110/` — 基线 L2 预测
- `PR-ENDO-data/P1_download/{nonl2,l2}/` — P1 预测

改数字只改 `data/key_numbers.json`，不要改脚本里的常量。

## 配色

图表用的是三个经过 CVD（色觉障碍）校验的分类色：

| 角色 | 色值 |
|---|---|
| P1 修复后 | `#2a78d6` 蓝 |
| 原始 PR-ENDO | `#eb6834` 橙 |
| Unity 真值 / 物理参考 | `#1baf7a` 绿 |

全对校验通过（最差 CVD ΔE 9.2，正常视觉 ΔE 24.0）。所有图都带直接数值标注，
不依赖颜色单独传达信息。中文字形用 Noto Sans CJK。

## 待补的东西

| 项 | 状态 | 找谁 |
|---|---|---|
| **重光照 GUI** | ⚠️ **不在本机的任何工作区里**（三个目录都搜过，无 tkinter/Qt/gradio/streamlit 代码）。需要录屏或源码 | 做这个 GUI 的组员 |
| **VR-Caps 渲染耗时** | ⚠️ 没有测量值。`fig01` 里路线 A 的"慢"目前只有定性说法 | VR-Caps 组员：863 张 B2 导出的秒/帧 |
| EndoGSLAM GUI 录屏 | 未录，但 `demo/clips/` 里有三段现成视频顶着 | 自己 |
| P3 / P6 实验 | 未做，已在总结里写明 | — |

拿到 VR-Caps 的耗时数字后，填进 `data/key_numbers.json` 的 `efficiency.unity_render_time`，
再跑一次 `make_figures.py` 就行。

## 关于体积

这个目录约 29 MB，其中 `gifs/` 占 17 MB、`figures/` 占 9.4 MB。
如果不想把二进制文件提交进仓库，在 `Group-Project/.gitignore` 里加：

```
presentation/gifs/
presentation/demo/clips/
```

文档和脚本都很小，值得提交——图随时能重新生成。
