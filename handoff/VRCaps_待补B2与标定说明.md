# 给 VRCaps 组员：待补 B2 数据与标定说明

## 结论

- **A 训练数据**：已收到 `VRCaps_lumen_100_v2`。
- **B1 光照参数**：已收到 `VRCaps_lumen_100_v2_relight_config`（`lights.json`、`transforms_relight.json`、`export_settings.json`）。
- **B2 多光照 GT + 深度**：**尚未收到** `relight/gt/` 和 `relight/depth/`。
- 按《对接说明：给负责 VR-Caps 的组员》第 2 节，**完整 B2** 是 VRCaps 组员的明确交付项，**只交给 Yutong**，不交给 PRENDO。
- PRENDO 渲染不需要完整 B2；PRENDO 需要的是 **聚光衰减指数 `n`**。
- `n` 由 Yutong 用 B2 的标定帧拟合；如果 PRENDO 侧代跑 `fit_spot_power.py`，只需要 B2 的 **15 个标定 EXR**，仍然不需要完整 B2。

## 1. 对接说明的要求

| 数据 | 内容 | 交付对象 | 状态 |
| --- | --- | --- | --- |
| A | 训练 RGB、深度、位姿 | PRENDO 组员 | 已收 |
| B1 | 选帧、光照配置、每帧光源位姿 | PRENDO + Yutong | 已收 |
| B2 | 多光照 GT 图像 + 深度（完整） | **只给 Yutong** | **缺失** |
| B2 标定子集 | 5 帧 × L0/L2_fwd + depth，共 15 个 EXR | Yutong，或 PRENDO 侧代拟合 `n` | **缺失** |

15 个标定 EXR 是 B2 的子集，只用于拟合 `n`；它不改变“完整 B2 只给 Yutong”的约定。

B2 的目录格式（对接说明第 4、6.4 节）：

```text
VRCaps_lumen_100_v2/relight/
├── gt/
│   └── <配置编号>/
│       └── <noshadow 或 shadow>/
│           └── <帧号>.exr       # 线性 RGB，half 或 single float
└── depth/
    └── <帧号>.exr               # 单通道线性 z 深度，无表面像素填 0
```

## 2. 最小解锁：15 个标定 EXR

`fit_spot_power.py` 只需要 `split=calib` 的 5 个标定帧。请先渲染：

```text
relight/gt/L0/noshadow/0001.exr
relight/gt/L0/noshadow/0032.exr
relight/gt/L0/noshadow/0050.exr
relight/gt/L0/noshadow/0067.exr
relight/gt/L0/noshadow/0097.exr

relight/gt/L2_fwd/noshadow/0001.exr
relight/gt/L2_fwd/noshadow/0032.exr
relight/gt/L2_fwd/noshadow/0050.exr
relight/gt/L2_fwd/noshadow/0067.exr
relight/gt/L2_fwd/noshadow/0097.exr

relight/depth/0001.exr
relight/depth/0032.exr
relight/depth/0050.exr
relight/depth/0067.exr
relight/depth/0097.exr
```

共 **10 张 GT EXR + 5 张深度 EXR**。

要求：

- `L0` 和 `L2_fwd` 必须在**同一曝光、同一光强**下渲染；
- `L2_fwd` 用 B1 的 `L2_fwd` 配置：外锥半角 30°（Unity 全角 60°），内锥半角 0°（HDRP innerSpotPercent=0）；
- 两个配置都用 `noshadow`；
- 5 个标定帧在 B1 里都是 `valid=true`；
- 线性 RGB EXR，不做色调映射/后处理；
- 深度为线性视空间 z，float，无表面像素填 0。

这 15 个文件交给 Yutong 后，由 Yutong 运行（如果 PRENDO 侧代跑，也同样需要这 15 个文件；完整 B2 仍只给 Yutong）：

```bash
cd Group-Project/relight_eval   # 或你本地仓库的对应路径
.venv/bin/python fit_spot_power.py /path/to/VRCaps_lumen_100_v2
```

得到 `eval_relight/VRCaps_lumen_100_v2/calibration.json` 的 `spot_power`，即 `n`。先检查 `spot_fit.png` 的拟合曲线是否贴合；确认后把 `n` 发给 PRENDO 组员。

## 3. 完整 B2（Yutong 对比用）

最小 15 个文件只用于拟合 `n`；正式对比仍需要完整 B2：

- 49 个 relight 相机帧的 `relight/depth/<帧号>.exr`；
- 每个 `valid=true` 的 `(config, frame, variant)` 对应 `relight/gt/<config>/<variant>/<frame>.exr`。

按 B1 的 `lights.json`：

- 名义 1015 张 variant 图像；
- 去掉 6 个 main/sequence 重叠的 L0 重复 → 1009；
- 73 个无效 L3 条目 × 2 variant = 146 张不渲染；
- **有效 B2 variant 路径为 863 张**。

`L3_u0.50`、`L3_r1.00` 没有有效条目；`L3_r0.50` 只有 17 个有效帧。请只渲染 `valid=true` 的条目，并与 B1 的 `frames` 一一对应。

## 4. 渲染设置与自检

按《对接说明_VRCaps组员》第 7 节：

- Linear 颜色空间，固定曝光，关闭自动曝光；
- 关闭色调映射、色彩校正、泛光、暗角、景深、运动模糊等后处理；
- 环境光/天空盒强度 0，关闭 GI、反射探针、SSR、SSAO；
- 每次渲染只有一个光源；
- `noshadow` 关闭阴影；`shadow` 只用于 L3/L5，硬阴影最高分辨率；
- 输出线性 float EXR，不要屏幕截图；
- 深度为线性视空间 z。

交付前至少检查：

- L1 线性关系、L4 颜色倍数（完整 B2）；
- `L0` / `L2_fwd` 标定帧的曝光和光强一致；
- 训练 PNG 与线性 EXR 编码一致；
- 过曝像素 < 1%（仅针对 **A 训练 PNG**；B2 线性 EXR 见下）；
- L3 的 `shadow` 不比 `noshadow` 亮；
- `lights.json` 每个 `valid=true` 条目都有对应 EXR；
- 帧号、相机矩阵与 `transforms_relight.json` 一致；
- `export_settings.json` 填写完整，并更新 B2 的实际设置（阴影、聚光角、深度通道等）。

### 关于高光/过曝

- **`过曝 < 1%` 只适用于 A 训练 PNG**（8-bit + clamp）。B2 的 `relight/gt/*.exr` 是线性 HDR，高光 >1 是正常物理现象，不要 clamp、不要 tonemap、不要用训练 PNG 的编码。
- 如果**把 B2 EXR 转成 PNG 预览**时看到高光截断，这是预览转换造成的，不影响 EXR 数据；交付请用 EXR。
- 如果 **EXR 本身** max=1 且大量像素=1，才是渲染/导出被 clamp，需要检查 AOV/EXR 导出。
- `fit_spot_power.py` 会排除饱和像素；`compare.py` 用线性 EXR。B2 中部分帧高光超过 1%（例如 L1_s2.0、L4）是预期的。
- V2 的 A 训练 PNG 已记录最大过曝比例 0.1189%，低于 1%；如果现在看到 25 张超过 1%，先确认它们是 B2 EXR 还是 A PNG。

## 5. 交付方式

- 完整 B2 目录交给 Yutong；PRENDO 侧渲染只需要 B1 和 A，不需要 B2 图像。
- 如果 PRENDO 侧代拟合 `n`，额外需要 §2 的 15 个标定文件；这不是完整 B2，也不改变完整 B2 只给 Yutong 的约定。
- 如果先交 15 个标定 EXR，请放在同一个 `VRCaps_lumen_100_v2/relight/` 下，保留 B1 的 `lights.json`、`transforms_relight.json`、`export_settings.json`。
- 重新导出请换新版本号，不要覆盖旧目录；如果只改 `relight/`，在交付说明里注明训练数据未变。

## 6. 为什么 PRENDO 需要 `n`

PRENDO 的 L2 聚光灯按 `cos(θ)^n` 衰减；`n` 必须来自 Unity 聚光灯的真实 falloff。B1 里 `spot_power_n` 是 `null`，不能用猜测值。拿到 `n` 后 PRENDO 才能补 L2 的 280 张，把当前 391 张补成完整的 671 张。
