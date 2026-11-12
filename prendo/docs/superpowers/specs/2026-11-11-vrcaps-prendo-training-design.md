# VRCaps + PR-ENDO 训练设计

- 日期：2026-11-11
- 状态：设计已确认，HPC 连接已打通，待实现
- 仓库：`SanoScience/PR-ENDO`，基线 commit `ee5362817ccc944292ae43aa9f667386a5ebc069`
- 工作分支：`vrcaps-reader`（`main` 保持与上游一致）
- HPC：Imperial College London HPC，CX3，用户名 `yl15425`
- 数据：根目录 `FOR_TEAM_VRCaps.zip` 内的 `VRCaps_lumen_100/`

## 1. 背景与目标

本项目研究不同光照条件对高斯泼溅重建的影响，计划用 PR-ENDO 做重光照。正式应用前，需要先在 VRCaps 合成场景上完成一次 PR-ENDO 训练，作为后续量化"PR-ENDO 重光照 vs Unity 真实重光照"的模型基础。

本阶段只关注**训练**，目标是：

1. 校验 `VRCaps_lumen_100` 与 PR-ENDO 的接口兼容性。
2. 在 Imperial HPC 上搭建可复现的 conda + CUDA 扩展环境。
3. 用官方 RotateColon 配方完成一次 30k 训练。
4. 产出后续重光照评估所需的全部模型状态、渲染、日志和元数据。

首轮训练本身不产生重光照结论；它的产出必须足以支撑下一阶段的光照对比实验。

## 2. 非目标

本阶段明确不做：

- 不设计或执行重光照评测协议。
- 不在 Unity 中重渲染 GT，不比较 PR-ENDO 重光照与 Unity 光照。
- 不跑 C3VD 或 RotateColon 原始数据。
- 不修改 PR-ENDO 的训练循环、损失函数、相机或渲染代码。
- 不应用 optional mask companion。
- 不做多光源、彩色光、光源平移等扩展实验。
- 不调超参搜参；只跑一套官方配方。

## 3. 数据与接口事实

来自 `FOR_TEAM_VRCaps.zip` 的 `START_HERE.md`、`README.md`、`VRCaps_lumen_100/README.md`、`REPRODUCTION.md` 与验证记录。

### 3.1 数据内容

| 项目 | 值 |
|---|---|
| 场景 | `VRCaps_lumen_100` |
| 训练帧 | 88 |
| 测试帧 | 12，索引 7,15,23,31,39,47,55,63,71,79,87,95 |
| 分辨率 | 1350 × 1080 |
| 训练 RGB | `train_views/0000.png` …（88 张） |
| 训练深度 | `depth_train/0000.tiff` …（88 张，float32，相机-Z，Unity 世界单位） |
| 测试 RGB | `test_views_1/0007.png` …（12 张，无 GT 深度） |
| 相机 | `transforms.json` / `transforms_test.json`，OpenGL camera-to-world，实际针孔内参 `fl_x=fl_y=664.663…`, `cx=675`, `cy=540`, `camera_angle_x=1.586227683125316` |
| 初始化点云 | `init_point_cloud.ply`，300,000 个 XYZ 点，仅来自训练帧 |
| 配置 | `prendo_config.json`：`depth_scale=1.0`, `depth_channel="ViewLayer.Depth.Z"`, `test_dir="test_views_1"`, `use_depth_mask=true`, `training_mask_dir="sidecars/minimal_training_masks"` |
| 单位 | 原始深度、平移、XYZ 使用一致的任意 Unity 世界单位；未验证物理毫米 |
| 测试深度 | reader 返回的是训练深度的内部拷贝，不是测试 GT，不参与 loss |

### 3.2 接口兼容性结论

- 原版 PR-ENDO 只认 `C3VD` 和 `Rotate` 两种数据，VRCaps 路径会抛 `NotImplementedError`。
- C3VD 路线不适用的原因：`readColonSceneInfo` 依赖 `C3VD_endogslam_optimized/params.npz`、EndoGSLAM 位姿和一个已训练的 Gaussian checkpoint，不是自包含格式。
- VRCaps 数据采用 RotateColon 布局，并由数据提供方提供一个最小 reader patch。
- 必须应用 `optional_mask_support/receiving_team_original.patch`：
  - `scene/dataset_readers.py`：新增 `readVRCapsSceneInfo()` 及辅助函数。
  - `scene/__init__.py`：新增 VRCaps dispatch 分支。
- 打 patch 后，数据路径中不能出现 `C3VD`，否则会命中 C3VD 分支。
- reader 的 `depth_scale` 来自 `prendo_config.json`，最终深度 target = `raw_depth / depth_scale` = `raw_depth`，与位姿和点云同一单位。
- reader 忽略 `use_depth_mask` / `training_mask_dir`；本阶段不应用 mask companion，因此 mask 监督不生效。minimal mask 只排除 956/128,304,000 像素（0.000745%），对首轮训练可忽略。
- 数据提供方**没有**运行过完整 Scene/Gaussian 加载和训练；本阶段是第一次真实运行。

### 3.3 关键参数

| 参数 | 值 | 来源 |
|---|---|---|
| `--resolution` | 1 | 数据提供方要求，保持 1350×1080 和内参 |
| `--iteration` | 30000 | 官方 RotateColon 命令 |
| `--eval` | 开 | 使用显式 12 个测试帧 |
| `--K_normals` | 30 | 官方 RotateColon 命令 |
| `--diffuse_loss_weight` | 0.01 | 官方 RotateColon 命令 |
| `--lambda_depth` | 0.2 | 官方 RotateColon 命令 |
| `--lambda_norm` | 0.1 | 官方 RotateColon 命令 |
| `--albedo_loss_weight` | 1.0 | 官方 RotateColon 命令 |
| `--densification_interval` | 1000 | 官方 RotateColon 命令 |
| `--opacity_reset_interval` | 3000 | 官方 RotateColon 命令 |
| `--densify_from_iter` | 1000 | 官方 RotateColon 命令 |
| `--densify_until_iter` | 10000 | 官方 RotateColon 命令 |
| `--mlp_lr` / `--grid_lr` | 0.0005 / 0.0005 | 官方 RotateColon 命令 |
| `--end_diffuse_loss_iter` | 20000 | 官方 RotateColon 命令 |
| `--use_hg` | 关闭 | 官方默认 |
| `depth_scale` | 1.0 | 数据提供方配置，不修改 |

## 4. 关键决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 接口 | 只应用官方 reader patch，2 个文件，提交到 `vrcaps-reader` 分支 | 数据提供方已验证；`main` 保持与上游一致；改动可审计、可回退 |
| 数据格式 | 保持 `VRCaps_lumen_100` 原样，`depth_scale=1.0` | 深度、位姿、点云单位一致；不伪造 C3VD/EndoGSLAM 状态 |
| 训练配方 | 官方 RotateColon 30k 命令 | 数据布局、帧数、合成深度与该配方对应 |
| mask | 不应用 companion | 只影响 956 个像素；避免修改训练代码 |
| 环境 | miniforge + Python 3.10 + PyTorch 2.1.0/cu121 + 源码编译 CUDA 扩展 | 与官方 README 对齐，避免 HPC module 版本漂移 |
| 存储 | 代码/环境/最终产物在 `$HOME`，数据和训练输出在 ephemeral 区 | `$HOME` 持久；ephemeral 容量大但 30 天清理。登录节点未导出 `$EPHEMERAL`，实际路径为 `/rds/general/user/yl15425/ephemeral` |
| HPC 调度 | PBSPro（不是 Slurm），GPU 用 `ngpus=1:gpu_type=L40S` | 官方文档；L40S 48 GB 满足 30 GB 需求 |

## 5. 系统组成

本阶段新增文件（不修改官方训练代码）：

```text
docs/superpowers/specs/2026-11-11-vrcaps-prendo-training-design.md
patches/vrcaps_reader.patch                 # 从 zip 复制的 official reader patch
tools/validate_vrcaps_interface.py          # 接口/深度/投影/GPU smoke 校验
tools/collect_artifacts.py                  # manifest 和产出一致性检查
hpc/env.sh                                  # module/conda/CUDA 环境
hpc/env_build.pbs                           # GPU 节点编译 CUDA 扩展
hpc/smoke.pbs                               # GPU Scene/render smoke
hpc/train.pbs                               # 30k 训练
hpc/render.pbs                              # 训练后 render + light rotation + metrics
```

工作分支上的提交顺序：

1. `docs: add VRCaps PR-ENDO training design`
2. `chore: add official VRCaps reader patch`
3. `tools: add VRCaps interface validation`
4. `hpc: add PBS environment and training scripts`
5. `tools: add artifact collection`

`main` 不产生任何提交。

## 6. 执行流程

### 6.1 本地准备

1. 创建并切换到 `vrcaps-reader` 分支。
2. 从 `FOR_TEAM_VRCaps.zip` 提取 `optional_mask_support/receiving_team_original.patch` 到 `patches/vrcaps_reader.patch`，记录 sha256。
3. 解压数据集到仓库外的 staging 目录，例如 `/home/yiping/Documents/my_repo/PR-ENDO-data/VRCaps_lumen_100`。任何父目录名不得含 `C3VD`。
4. 校验压缩包与解压结果：88/88/12 文件数、命名、TIFF dtype、PNG 尺寸、JSON 必需字段、PLY 顶点数。本机没有 PIL/numpy，像素级校验在 HPC 环境内完成；本地只做文件级校验。
5. `git apply -p0 --check patches/vrcaps_reader.patch`，然后在工作区应用；不提交官方源码修改，patch 文件本身提交。patch 的路径头是 `scene/...`，必须用 `-p0`，否则 `git apply` 默认 `-p1` 会剥掉 `scene/`。

### 6.2 HPC 环境

1. 用户执行一次持久 SSH 登录：

```bash
ssh -M -S ~/.ssh/hpc.sock yl15425@login.cx3.hpc.imperial.ac.uk
```

后续命令复用：

```bash
ssh -S ~/.ssh/hpc.sock yl15425@login.cx3.hpc.imperial.ac.uk '<command>'
rsync -e "ssh -S ~/.ssh/hpc.sock" ...
```

2. 确认存储变量和空间。登录节点默认没有 `$EPHEMERAL`，显式设置：

```bash
export EPHEMERAL=/rds/general/user/yl15425/ephemeral
echo "$HOME"; echo "$EPHEMERAL"; ls -ld "$HOME" "$EPHEMERAL"; df -h "$HOME" "$EPHEMERAL"
```

3. 在 `$HOME/prendo/code/PR-ENDO` 克隆仓库，checkout `ee53628`，再切到 `vrcaps-reader` 或上传本地分支；初始化 submodules。
4. 加载 miniforge：

```bash
module load miniforge/3
miniforge-setup
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda config --remove channels defaults || true
conda config --add channels conda-forge
conda config --set auto_activate_base false
```

5. 创建环境：

```bash
conda create -n prendo python=3.10 -y
conda activate prendo
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install tqdm lpips plyfile matplotlib natsort pytorch_ssim \
  pytorch_msssim "imageio[ffmpeg]" openexr
```

6. 在 GPU 节点编译扩展（`hpc/env_build.pbs`）：

```bash
module load CUDA/12.1.1   # 以 module avail CUDA 的实际版本为准
export TORCH_CUDA_ARCH_LIST=8.9   # L40S=Ada sm_89；若用 A100 则改为 8.0
pip install submodules/depth-diff-gaussian-rasterization/
pip install submodules/simple-knn/
pip install submodules/gaussian-norms/
pip install https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

7. 验证：

```bash
python -c "import torch, tinycudann, diff_gaussian_rasterization, simple_knn, faiss; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

预期：PyTorch 2.1.0、CUDA 12.1、L40S；所有扩展可导入。

### 6.3 接口校验

在 HPC 的项目环境中运行 `tools/validate_vrcaps_interface.py`：

1. 解析 `transforms.json` / `transforms_test.json`：88/12 帧、必需字段、c2w 底行、旋转行列式。
2. 校验内参：`fl_x/fl_y/cx/cy` 与 `camera_angle_x`、图像尺寸一致。
3. 读取 3 个训练 PNG 和 TIFF：尺寸 1350×1080、TIFF float32、正值、无 NaN；打印 min/max/分位数。
4. 调用 patched `readVRCapsSceneInfo`：train 88 / test 12；`Camera.original_depth` 与原始 TIFF 的相对误差 < 1e-5；测试相机深度不参与 loss。
5. 将 300k PLY 点重投影到 3 个训练视角，最大误差 < 0.01 px。
6. GPU smoke：实例化 `Scene`、`GaussianModel`，渲染一个训练视角，计算 loss 并反向传播；记录 `torch.cuda.max_memory_allocated()`。
7. 若峰值显存 > 40 GB：训练改用 `--resolution 2`，并把该变更记录为一次独立实验配置；否则保持 `--resolution 1`。

### 6.4 训练

PBS 作业头：

```bash
#!/bin/bash
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=24:00:00
#PBS -N prendo_vrcaps
#PBS -j oe

cd ${PBS_O_WORKDIR}
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate prendo
```

训练命令：

```bash
python train_viewdir.py \
  --model_path $EPHEMERAL/prendo/runs/vrcaps_lumen_100_r30k \
  -s $EPHEMERAL/prendo/data/VRCaps_lumen_100 \
  --eval --resolution 1 --iteration 30000 \
  --lambda_depth 0.2 --lambda_norm 0.1 --albedo_loss_weight 1.0 \
  --densification_interval 1000 --opacity_reset_interval 3000 \
  --densify_until_iter 10000 --densify_from_iter 1000 \
  --mlp_lr 0.0005 --grid_lr 0.0005 --end_diffuse_loss_iter 20000 \
  --port 6000 --K_normals 30 --diffuse_loss_weight 0.01
```

- 训练前记录 `nvidia-smi`、`torch.cuda.get_device_name`、`CUDA_VISIBLE_DEVICES`。
- 训练日志写 `$EPHEMERAL/prendo/runs/vrcaps_lumen_100_r30k/train.log`。
- 默认在 1k/10k/20k/30k 保存点云和 checkpoint；NaN 时脚本写 `LOSS NAN` 并保存当前 checkpoint 后退出。
- 若作业被排队/抢占或 walltime 不够：用 `--start_checkpoint <chkpnt>` 从最近 checkpoint 续跑。
- 若 OOM：改 `--resolution 2`，作为独立配置记录；不改其它参数。
- 若 `LOSS NAN`：保留 checkpoint 和日志，先诊断深度/法线/loss 尺度，再决定是否调 `lambda_depth`；任何参数变更都作为独立实验记录。

### 6.5 训练后渲染与打包

在同一个 `prendo` 环境中运行：

```bash
python render.py --model_path $EPHEMERAL/prendo/runs/vrcaps_lumen_100_r30k \
  --eval -s $EPHEMERAL/prendo/data/VRCaps_lumen_100 \
  --skip_train --iteration 30000

python render_light_rotations.py --iteration 30000 \
  --model_path $EPHEMERAL/prendo/runs/vrcaps_lumen_100_r30k \
  --eval -s $EPHEMERAL/prendo/data/VRCaps_lumen_100 \
  --spotlight_attenuation_power 10 --max_rot 12

python metrics.py $EPHEMERAL/prendo/runs
```

`render.py` 产出测试集 render / GT / albedo / normals / disable_reflections 和视频；`render_light_rotations.py` 产出 learned light 的光照旋转视频；`metrics.py` 在 runs 目录写 `mean_results.txt`。

`tools/collect_artifacts.py` 生成 `run_manifest.json`，字段见附录 B，并对关键文件计算 sha256。

最终打包（在 HPC 上执行）：

```bash
cd $EPHEMERAL/prendo/runs
tar -czf vrcaps_lumen_100_r30k_bundle.tar.gz vrcaps_lumen_100_r30k
mkdir -p $HOME/prendo/artifacts
cp vrcaps_lumen_100_r30k_bundle.tar.gz $HOME/prendo/artifacts/
sha256sum $HOME/prendo/artifacts/vrcaps_lumen_100_r30k_bundle.tar.gz
```

下载到本地（在笔记本上执行）：

```bash
rsync -e "ssh -S ~/.ssh/hpc.sock" -av \
  yl15425@login.cx3.hpc.imperial.ac.uk:'$HOME/prendo/artifacts/vrcaps_lumen_100_r30k_bundle.tar.gz' \
  ./artifacts/
```

`$EPHEMERAL` 30 天清理，不能作为唯一副本；`$HOME` 和本地各保留一份。

## 7. 产出清单

训练目录 `vrcaps_lumen_100_r30k/` 必须包含：

```text
point_cloud/iteration_30000/point_cloud.ply
chkpnt30000.pth
chkpnt_mlp30000.pth
chkpnt_grid_camera30000.pth
chkpnt_light30000.pth
cfg_args
cameras.json
results.txt
train.log
events.out.tfevents.*              # tensorboard
test/ours_30000/renders/*.png
test/ours_30000/gt/*.png
test/ours_30000/albedo/*.png
test/ours_30000/normals/*.png
test/ours_30000/disable_reflections/*.png
rotate_light_video_maxrot12_cospower10/*.mp4
run_manifest.json
```

保留 10k/20k 的 checkpoint 和 point cloud 作为中间状态；1k 可按空间决定是否保留。所有这些是后续重光照评估的输入，不能只保留最终 PLY。

## 8. 验收标准

1. 本地文件级校验和 HPC 接口校验全部通过。
2. GPU smoke 通过，峰值显存 < 40 GB（否则按 6.3 第 7 条切换 resolution 2 并记录）。
3. 训练 30k 完成，日志无 `LOSS NAN`，`results.txt` 有最终 test 指标。
4. `render.py` 和 `render_light_rotations.py` 能从磁盘 checkpoint 重跑成功。
5. `run_manifest.json` 包含附录 B 的全部字段，关键文件 sha256 可复核。
6. 产出包下载到本地后能解压，文件数量/大小与 manifest 一致。

## 9. 风险与回退

| 风险 | 处理 |
|---|---|
| 1350×1080 OOM | 先测 GPU smoke 峰值；超过 40 GB 改用 `--resolution 2`，记录为独立配置 |
| tiny-cuda-nn / rasterizer 编译失败 | 确认 CUDA module 与 PyTorch cu121 匹配；设置 `TORCH_CUDA_ARCH_LIST=8.9`；必要时用 conda `nvidia` channel 安装 `cuda-toolkit=12.1` |
| submodule 克隆失败（HPC 网络限制） | 在本地 `git submodule update --init --recursive` 后，把三个 submodule 目录打包上传，再在 HPC 上从本地目录 `pip install` |
| `openexr` pip 包缺系统库 | 改用 conda-forge 的 `openexr`，或安装 `libopenexr-dev`；数据本身是 TIFF，OpenEXR 仅为模块级 import |
| 30k 欠拟合 | 保留日志和 checkpoint，补 40k / `K_normals=100` 作为独立实验 |
| 深度损失尺度偏大 | 打印深度分位数；确认 `delta=0.2` 的合理性；必要时调 `lambda_depth` 并单独记录 |
| GPU 排队久 | 先用短 smoke 作业验证；训练作业 24h walltime；必要时 checkpoint 续跑 |
| `$EPHEMERAL` 30 天清理 | 训练后立即复制到 `$HOME` 和本地 |
| 登录节点资源限制 | 编译、smoke、训练全部在 GPU 节点 PBS 作业中执行 |
| 路径含 `C3VD` | 部署前 fail-fast 检查；staging 和 HPC 路径都不含该子串 |

## 10. 运维信息

| 项目 | 值 |
|---|---|
| HPC 用户 | `yl15425` |
| 集群 | CX3，`login.cx3.hpc.imperial.ac.uk`（本次登录落到 `login-ai`） |
| 调度器 | PBSPro |
| GPU 队列 | `v1_gpu72`，GPU 用 `ngpus=1:gpu_type=L40S`；A100 40 GB 为备选 |
| 已确认的 module | `CUDA/12.1.1`、`miniforge/3` 均可用 |
| 代码/环境 | `$HOME/prendo/code/PR-ENDO`, `$HOME/prendo/envs/prendo` |
| 数据 | `/rds/general/user/yl15425/ephemeral/prendo/data/VRCaps_lumen_100` |
| 训练输出 | `/rds/general/user/yl15425/ephemeral/prendo/runs/vrcaps_lumen_100_r30k` |
| 最终产物 | `$HOME/prendo/artifacts/` + 本地下载 |
| SSH 复用 | `~/.ssh/hpc.sock`（已验证 `hostname` 返回 `login-ai`） |

## 11. 未测量项与决策规则

这些不是待定需求，而是执行中必须测量并按规则处理的项目：

1. **深度范围**：在 6.3 打印 min/max/p1/p50/p99。若 p99 小于 0.2 或大于 100，记录并复核 `lambda_depth=0.2`、Huber `delta=0.2` 的合理性，但不自动改参数。
2. **峰值显存**：在 6.3 的 GPU smoke 测量。大于 40 GB 则训练用 `--resolution 2`；否则用 `--resolution 1`。
3. **CUDA module 版本**：已确认 `CUDA/12.1.1` 可用，直接使用；若编译异常再用 conda `nvidia` channel 的 `cuda-toolkit=12.1`，并记录实际版本。
4. **队列 walltime 上限**：qsub 时以集群实际限制为准。若上限小于 24h，改为允许的最大值并依赖 checkpoint 续跑。
5. **训练耗时**：记录 walltime 和每 1000 步耗时，用于后续评估阶段的预算。

## 附录 A：完整环境检查命令

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("mem GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
import tinycudann, diff_gaussian_rasterization, simple_knn, faiss
print("extensions ok")
PY
```

## 附录 B：`run_manifest.json` 字段

下面是结构示例；`...`、`0`、`null` 是待运行时填入的字段占位，不是设计缺口。`outputs` 至少列出最终 PLY、四个 checkpoint 和 `results.txt`，渲染目录与日志也要在打包清单中逐项记录 sha256。

```json
{
  "run_id": "vrcaps_lumen_100_r30k",
  "timestamp_utc": "...",
  "repo": "SanoScience/PR-ENDO",
  "base_commit": "ee5362817ccc944292ae43aa9f667386a5ebc069",
  "branch": "vrcaps-reader",
  "reader_patch_sha256": "...",
  "dataset": {
    "name": "VRCaps_lumen_100",
    "train_count": 88,
    "test_count": 12,
    "test_indices": [7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95],
    "depth_scale": 1.0,
    "transforms_sha256": "...",
    "transforms_test_sha256": "...",
    "init_point_cloud_sha256": "...",
    "config_sha256": "..."
  },
  "environment": {
    "python": "3.10.x",
    "torch": "2.1.0",
    "cuda": "12.1",
    "conda_env": "prendo",
    "pip_freeze": "environment/pip_freeze.txt",
    "conda_export": "environment/conda_export.txt"
  },
  "hardware": {
    "gpu": "NVIDIA L40S",
    "gpu_memory_gb": 48,
    "pbs_job_id": "...",
    "pbs_resources": "select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=L40S"
  },
  "command": "python train_viewdir.py ...",
  "iterations": 30000,
  "resolution": 1,
  "walltime_seconds": 0,
  "peak_vram_gb": 0,
  "metrics": {
    "test_psnr": null,
    "test_ssim": null,
    "test_lpips": null,
    "test_mssim": null
  },
  "outputs": [
    {"path": "point_cloud/iteration_30000/point_cloud.ply", "sha256": "..."},
    {"path": "chkpnt30000.pth", "sha256": "..."},
    {"path": "chkpnt_mlp30000.pth", "sha256": "..."},
    {"path": "chkpnt_grid_camera30000.pth", "sha256": "..."},
    {"path": "chkpnt_light30000.pth", "sha256": "..."}
  ],
  "notes": "Mask companion not applied; reader ignores mask keys. Test depth is an internal placeholder."
}
```
