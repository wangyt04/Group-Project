# L2 4× relight evaluation — run 20261113-0110

## 摘要

- 新 L2：**200,000 cd（4× 基准光强）**，外锥 **全角 160°（半角 80°）**，内锥 **半角 10°（全角 20°）**，
  `innerSpotPercent = 12.5`，7 个方向、220 条预测。
- 评估集：新 B2 L2 GT（30 个 test 相机 × 7 方向 = 210 个 test 比较；5 个 calib 帧仅作支持数据）。
- **PSNR 32.99 dB**（7 个 config 平均），比未重光照的 L0 基线 **+11.56 dB**；
  **SSIM 0.9863**，**LPIPS 0.0823**（max-side 512），**ratio_corr 0.993**，**ratio_rmse 0.068**。
- 7 个 config 全部 `effective = trusted = True`；方向一致性检查 **100% 通过**。
- **没有重新训练**：仍使用同一个 30k V2 模型（PSNR 34.787 / SSIM 0.9886 / LPIPS 0.0580）。

## 评估设置

| 项目 | 内容 |
|---|---|
| PRENDO 预测 | `pred_VRCaps_lumen_100_v2_20261113-0110`（220 条，PNG + EXR） |
| GT | `VRCaps_lumen_100_v2_B2_L2_4x_Outer160_InnerHalf10_GT_and_Parameters.zip` |
| L0 支持数据 | L0 GT 来自原始 B2（50k）；PRENDO L0 渲染复用旧 run `20261112-0332`（同模型、同数据） |
| 对应对齐 | `lights.json` 220 条 ↔ 新 L2 GT 220 个 EXR ↔ 预测 220 PNG/EXR，零缺零多；L0 支持数据严格裁剪为 35 帧 |
| 标定 | `prendo/runs/20261113-0110/calibration/calibration.json`：n = 2.4004990755616116，RMSE 0.02047 |
| 评估代码 | `relight_eval/compare.py`（Group-Project `d6ad518`） |
| 指标定义 | 显示空间 PSNR/SSIM、线性空间 ratio 指标；有效像素 = depth > 0；ratio 指标额外排除饱和/过暗像素 |
| LPIPS | VGG，`max-side 512`（GPU） |

评估命令：

```bash
cd Group-Project/relight_eval
PYTHONPATH=/tmp/np126 python compare.py \
  /path/to/PR-ENDO-data/L2_4x_eval/VRCaps_lumen_100_v2 \
  /path/to/PR-ENDO-data/L2_4x_eval/pred \
  --calibration ../prendo/runs/20261113-0110/calibration/calibration.json \
  --out-root ../prendo/runs/20261113-0110/eval \
  --run L2_4x --lpips vgg --lpips-max-side 512
```

## 每 config 结果（30 个 test 帧/配置）

| config | PSNR↑ | SSIM↑ | LPIPS↓ | 未重光照 PSNR | PSNR 增益 | ratio_corr↑ | ratio_rmse↓ | ratio_bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L2_fwd | 32.57 | 0.9861 | 0.0834 | 21.19 | **+11.38** | 0.989 | 0.053 | −0.033 |
| L2_yaw+10 | 31.98 | 0.9859 | 0.0834 | 19.98 | **+12.00** | 0.996 | 0.053 | −0.033 |
| L2_yaw−10 | 33.58 | 0.9867 | 0.0825 | 22.67 | **+10.91** | 0.992 | 0.063 | −0.015 |
| L2_yaw+20 | 31.68 | 0.9853 | 0.0835 | 19.22 | **+12.46** | 0.998 | 0.053 | −0.027 |
| L2_yaw−20 | 34.67 | 0.9872 | 0.0799 | 23.06 | **+11.62** | 0.992 | 0.113 | +0.025 |
| L2_pitch+20 | 31.96 | 0.9859 | 0.0852 | 19.75 | **+12.21** | 0.998 | 0.053 | −0.019 |
| L2_pitch−20 | 34.46 | 0.9873 | 0.0784 | 24.09 | **+10.37** | 0.988 | 0.085 | +0.005 |
| **平均** | **32.99** | **0.9863** | **0.0823** | **21.42** | **+11.56** | **0.993** | **0.068** | **−0.014** |

参考：PRENDO 在 L0 上的重建 PSNR 为 **34.23 dB**，因此 L2 重光照结果距离模型自身的
L0 重建上限约 1.2 dB；比“不重光照”的 L0 渲染高 10–12.5 dB。

## 检查结果

| 检查 | 结果 |
|---|---|
| `predictions_complete` | pass（220 条全部有预测） |
| `record_dataset` | pass（`VRCaps_lumen_100_v2`） |
| `spot_power` | pass（record 2.4004990755616116 vs calibration 2.4005） |
| `direction_L2_yaw±10/±20`、`L2_pitch±20` | 全部 pass，**同侧增亮 100%**，ratio_corr 0.99–1.00 |
| `effective` / `trusted`（每个 config） | 全部 True |

## 解读

1. **L2 的“太暗”问题已解决**：新 L2 在可见范围内是 4× 宽照明（内锥 10° 平顶 + 160° 外锥），
   PRENDO 渲染的亮度与 GT 同量级；相对 L0 基线 PSNR 提升 10–12.5 dB。
2. **方向性仍然正确**：虽然内锥很小、7 个方向差异主要来自几何与阴影，方向一致性检查仍 100%
   通过（ratio_corr 0.99–1.00），说明光束转向与 GT 一致。
3. **精度接近模型上限**：L2 PSNR 32.99 dB，仅比模型自身 L0 重建 34.23 dB 低约 1.2 dB；
   ratio_bias 平均 −0.014，整体亮度尺度匹配良好。
4. **近似与限制**：
   - PRENDO 用 `cos(θ)^2.4005` 近似 Unity 的“10° 平顶 + 160° rolloff”，实测 RMSE 0.020；
     外缘 >50° 的实测衰减比 `cos^n` 略快。
   - PRENDO 输出是 SDR（display 截断到 [0,1]）；B2 GT 是 HDR（L2 有 >1 的数值），
     比较按 `compare.py` 的 clip+gamma 处理，因此高光细节不参与亮度误差。
   - LPIPS 在 max-side 512 下计算；L0 支持预测来自旧 run（同一模型、同一数据，确定性渲染）。
5. **取代关系**：旧 L2（280 张、60° 全角、n=23.135）**已被本 run 取代**，不应再用于 L2 评估；
   `L2_yaw±40` 已由新 supplement 退役。旧 run 的 L0/L1/L3/L4/L5 仍有效。

## 产物

- 预测 bundle：`pred_VRCaps_lumen_100_v2_20261113-0110.tar.gz`
  sha256 `ae192089e286ba8aa6546d4d1ec91f59840e96afdacd7f136bd72be9231f0930`（1.0 GB，1,070,259,588 字节）
- 本地渲染数据：`PR-ENDO/artifacts/pred_VRCaps_lumen_100_v2_20261113-0110/`（220 PNG + 220 EXR）
- 评估输出：`prendo/runs/20261113-0110/eval/VRCaps_lumen_100_v2/L2_4x/`
  （`per_image.csv`、`summary.csv`、`summary_levels.csv`、`checks.json`、`curves/`、`panels/`）
- 标定：`prendo/runs/20261113-0110/calibration/`（`calibration.json`、`spot_fit.png`）
- 代码：`code_commit = https://github.com/wangyt04/Group-Project@43206c84356f43075dcd2d924e2fe46d7d650d93`
