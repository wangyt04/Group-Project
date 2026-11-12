# VRCaps 点光源重导出要求（PR-ENDO 重训用）

这份文件只针对**重新训练 PR-ENDO**所需的数据，不涉及 `relight/` 光照参数和多光照图像。等重训完成、模型检查通过后，再单独讨论 relight 数据。

## 1. 目标

沿用旧 `VRCaps_lumen_100` 的目录结构和相机/深度/点云，**只把训练和测试 RGB 重新渲染成“相机中心点光源”版本**，交付一个新版本文件夹，让 PR-ENDO 重新训练。

旧包是 HDRP 聚光灯（Spot，110°，偏移 `(0,0,-0.02)`，Range 2），不能用于最终重光照。新包必须是点光源。

## 2. 交付目录

新文件夹命名为：

```
VRCaps_lumen_100_v2/
├── transforms.json               # 复用旧包
├── transforms_test.json          # 复用旧包
├── train_views/0000.png          # 重新渲染：点光源
├── depth_train/0000.tiff         # 复用旧包
├── test_views_1/0007.png         # 重新渲染：点光源
├── init_point_cloud.ply          # 复用旧包（300k XYZ）
├── prendo_config.json            # 复用旧包
├── split_manifest.json           # 复用旧包，更新版本说明
├── light_settings.json           # 新增，记录训练光源
├── README.md                     # 更新版本与变更说明
└── sidecars/                     # 可选，复用旧包；PRENDO reader 不使用 mask
```

不要覆盖旧 `VRCaps_lumen_100`。所有父目录名不得包含 `C3VD`。

## 3. 必须改的：RGB 重新渲染

- `train_views/*.png`：88 张训练 RGB，**全部重渲染**。
- `test_views_1/*.png`：12 张测试 RGB，**全部重渲染**；测试帧也要用同一个点光源。
- 训练和测试使用**同一个相机轨迹、同一套相机参数、同一个点光源设置**。
- 图像分辨率、FOV、内参与旧包一致：1350×1080，`camera_angle_x`/`fl_x`/`fl_y`/`cx`/`cy` 不变，无畸变，主点在图像中心。
- 帧号、文件名、训练/测试划分与旧包完全一致：
  - 训练：除 7,15,23,31,39,47,55,63,71,79,87,95 以外的 88 帧
  - 测试：7,15,23,31,39,47,55,63,71,79,87,95

## 4. 训练光源设置（最关键）

| 项目 | 要求 |
| --- | --- |
| 光源类型 | **Point**，不是 Spot |
| 位置 | 相机中心，相机坐标偏移 **`(0, 0, 0)`** |
| 跟随相机 | 每帧刚性跟随相机；不是子物体也可以，脚本每帧更新变换 |
| 颜色 | 白色 `(1, 1, 1)` |
| 强度 | 固定，整个数据集一致；选择曝光使过曝像素 < 1% |
| Range | **≥ 20 × d_ref**；`d_ref` = 训练深度中位数 |
| 衰减 | 平方反比 |
| 阴影 | **关闭**（训练帧不要阴影） |
| 光源数量 | 每次渲染只有一个光源 |
| 环境光/天空盒 | 强度 0 |
| 间接光 | 关闭全局光照、反射探针、屏幕空间反射 |
| 色调映射 | 关闭 |
| 曝光 | 固定曝光，关闭自动曝光 |
| 其他后处理 | 全部关闭：泛光、暗角、色彩校正、胶片颗粒、景深、运动模糊 |
| 颜色空间 | Linear |
| 抗锯齿 | 可以开，所有渲染保持一致 |
| 材质 | 所有帧用同一套材质 |
| 图像输出 | 渲染到浮点纹理，按第 5 节转换为 PNG |
| 深度 | 复用旧包；如需重新导出，线性视空间 z，float32 TIFF，单位和位姿一致 |

**不要使用旧聚光灯设置**：Spot、110°、inner 1°/100%、偏移 -0.02、Range 2 都不适用于新训练数据。

## 5. 图像编码

- 线性图像先截断到 0–1；
- 取 `1/2.2` 次方；
- 存成 8 位 RGB PNG。
- 不要用 Unity 的 sRGB 曲线、色调映射或屏幕截图。
- 过曝（线性值 ≥ 1）的像素少于 1%。

## 6. 可以直接复用的旧包内容

只要场景几何和相机没有变化，下面这些直接复制旧包，不要重算：

- `transforms.json`、`transforms_test.json`
- `depth_train/*.tiff`（88 张 float32 深度）
- `init_point_cloud.ply`（300k XYZ）
- `prendo_config.json`（`depth_scale=1.0`、`test_dir=test_views_1`）
- `split_manifest.json`（更新版本/说明即可）
- `sidecars/`（可选；mask 不影响 PRENDO reader）

如果场景几何或相机发生了变化，**先告知 PR-ENDO 组员**，深度和初始点云需要重新生成并验证，不能直接复用。

## 7. 新增 `light_settings.json`

PRENDO reader 不读这个文件，但它用于训练记录和 `record.json`。示例：

```json
{
  "version": "VRCaps_lumen_100_v2",
  "unity_version": "2019.4.40f1",
  "render_pipeline": "HDRP 7.3.1",
  "color_space": "Linear",
  "tone_mapping": "off",
  "exposure": 0.0,
  "base_light": {
    "type": "point",
    "position": "camera_center",
    "camera_offset": [0.0, 0.0, 0.0],
    "color": [1.0, 1.0, 1.0],
    "intensity": 5.0,
    "range": 7.0,
    "shadows": false
  },
  "d_ref": 0.35,
  "depth_encoding": "linear view-space z, float32 TIFF, depth_scale=1.0",
  "resolution": [1350, 1080],
  "camera_angle_x": 1.586227683125316,
  "notes": "RGB re-rendered with a camera-centered point light; poses, depth, point cloud and split reused from VRCaps_lumen_100."
}
```

请按实际值填写；`d_ref` 和 `range` 必须真实计算，不要照抄示例。

## 8. 交付前自检

- [ ] 88 张训练 PNG + 12 张测试 PNG + 88 张深度 TIFF，数量和文件名正确。
- [ ] `transforms.json`/`transforms_test.json` 与旧包一致；帧号、位姿、内参没有变化。
- [ ] 所有 PNG 为 1350×1080、8 位 RGB。
- [ ] 所有深度 TIFF 为 float32、正值、无 NaN；`depth_scale=1.0` 下与位姿/点云单位一致。
- [ ] 训练和测试 RGB 都使用相机中心点光源；没有 Spot 锥、没有阴影、没有环境光/间接光。
- [ ] 过曝像素 < 1%。
- [ ] 初始点云 300k 点；把点反投影到训练帧，点和深度贴合，不散乱、不重影。
- [ ] 取相隔较远的两帧，把深度按位姿反投影到世界，同一表面重合。
- [ ] `light_settings.json` 填写完整，`d_ref`/`range`/强度/曝光与实际一致。
- [ ] 父目录名不含 `C3VD`。
- [ ] 交付一个完整的 `VRCaps_lumen_100_v2/`，不要只发图片。

## 9. 交付方式

- 新版本文件夹打包为 `VRCaps_lumen_100_v2.zip`，通过 OneDrive/共享盘发给 PR-ENDO 组员。
- 在 README 里写清楚：相对 v1 只重渲染了训练/测试 RGB，相机、深度、点云、划分复用。
- 如果后续还要改 `relight/`，另开版本号，不要覆盖本次训练数据。

## 10. PR-ENDO 侧会做什么

收到数据后：

1. 结构、位姿、深度、点云一致性检查；
2. 用现有 `vrcaps-reader` 加载，确认 88/12；
3. 30k 训练；
4. 训练帧/测试帧渲染与 PSNR/SSIM/LPIPS；
5. 如果指标正常，再讨论 relight 参数和多光照渲染。

如果结构检查或短训练不通过，会给出逐项回执，不会直接进入完整训练。
