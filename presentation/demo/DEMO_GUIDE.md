# 两个 GUI 的演示方案

## 结论先说

**两个 GUI 都录屏，现场放视频；不要在 PPT 里现场跑。** 现场只在 Q&A 被追问、而且本机已经
接好环境时，才切出去跑一次真的。

理由：

| 风险 | 说明 |
|---|---|
| 环境 | 两个 GUI 都要 CUDA + 编译好的高斯光栅化扩展。演讲用的机器/投影电脑大概率没有 |
| 显存 | 交互式真实泼溅渲染要把整个点云放进显存；共用的会议室机器不可靠 |
| 时间 | 加载 `params.npz`/PLY 要几秒到几十秒，台上等不起 |
| 可控 | 录屏可以剪掉加载、放慢关键操作、加字幕箭头；现场拖动条很难一次拖到想要的那一帧 |

录屏反而能做到现场做不到的事：**把"拖动一个参数 → 画面怎么变"这件事讲清楚**，这才是 GUI 要传达的东西。

---

## GUI 1：相机位姿调整 + 深度可视化（EndoGSLAM）

**位置**：`~/Documents/my_repo/EndoGSLAM/my_scripts/view_gaussian_interactive.py`
**环境**：`~/miniconda3/envs/endogslam`（已验证可以 `import diff_gaussian_rasterization`）

它做什么：加载重建好的高斯场景，用真正的泼溅光栅化器（不是 Open3D 的点云显示）实时渲染，
yaw / pitch / distance / zoom 四个拖动条控制轨道相机，按 `s` 把**当前这一帧**存成一整套训练样本：

| 文件 | 格式 | 用途 |
|---|---|---|
| `view_XXXX.png` | RGB8 | 新样本的图像 |
| `view_XXXX_depth.tiff` / `.npy` | float32 | 度量深度，NaN = 无效 |
| `view_XXXX_depth_16bit.png` | uint16 | 深度相机惯例，`depth = value / depth_scale` |
| `view_XXXX_depth_gray.png` / `_depth_vis.png` | uint8 | 预览（灰度 / TURBO 伪彩） |
| `view_XXXX_alpha.png` | uint8 | 累积 alpha，即有效性掩码 |
| `view_XXXX_w2c.txt` / `_K.txt` | 文本 | 外参 / 内参 |

**这就是"新样本生成"的完整证据**：一次按键 → 图像 + 四种格式的深度 + 相机标定，
正是下游训练需要的东西。

### 建议的录屏脚本（约 70 秒）

```bash
conda activate endogslam
cd ~/Documents/my_repo/EndoGSLAM
python my_scripts/view_gaussian_interactive.py experiments/C3VD_base/sigmoid_t3_a \
    --depth-range 0 100
```

1. **0–10 s** 窗口打开，停在初始视角。旁白：这是从一段真实 C3VD 结肠镜视频重建出来的高斯场景。
2. **10–30 s** 慢慢拖 yaw，再拖 pitch。**要拖到一个训练视频里根本没出现过的角度**——
   这是整个演示的重点：原始视频里相机只会往前走，这个角度是重建出来的。
3. **30–40 s** 按 `d` 切到深度显示，再按 `d` 切回。`--depth-range 0 100` 让颜色尺度固定
   （C3VD 深度单位是毫米，GT 范围 0–100），不加这个参数每一帧会各自重新归一化，看起来会跳。
4. **40–55 s** 按 `s` 保存。切到文件管理器或 `ls` 一下输出目录，把那一串文件展示出来。
5. **55–70 s** 打开 `view_XXXX_depth_vis.png` 和 `view_XXXX.png` 并排看一眼。

> ⚠️ 连续按 `s` 而不动相机会存出一模一样的副本（每份约 3 MB）。录屏时每次保存之间记得动一下相机。

### 备用素材（已经在 `clips/` 里，不需要重新跑）

| 文件 | 内容 |
|---|---|
| `clips/endogslam_scene_flythrough.mp4` | 场景穿行，展示重建的完整性 |
| `clips/endogslam_scene_turntable.mp4` | 环绕一周，展示这是真正的 3D 而不是一张图 |
| `clips/endogslam_gt_vs_render.mp4` | 真值 vs 渲染并排，展示重建精度 |

即使录屏来不及，这三段也够撑住"相机可以自由移动"这个论点。

---

## GUI 2：重光照参数调整（另一位组员）

**状态：⚠️ 这个工具不在本机的三个工作区里（`PR-ENDO` / `Group-Project` / `EndoGSLAM` 都搜过，
没有任何 tkinter / Qt / gradio / streamlit 的代码）。需要那位组员提供。**

### 需要他交付的东西

1. **一段录屏**（首选，60–90 秒），内容见下面的分镜；
2. 或者**源码 + 运行说明**，这样可以在本机重录；
3. 顺带确认它加载的是哪个模型——如果是基线（0.2 裁剪）模型，近壁会糊，录屏时**避开贴近肠壁的视角**，
   或者干脆换成 P1 模型（在 HPC `$EPHEMERAL/prendo/runs/vrcaps_lumen_100_v2_p1_nearcull_r30k/`）。

### 建议的分镜

这个 GUI 要讲的点和 GUI 1 不一样：**GUI 1 证明"相机能动"，GUI 2 证明"光能动"**。
顺序最好和我们的 L1–L4 层级对上，这样和评估结果能直接呼应：

| 时间 | 操作 | 对应层级 | 旁白要点 |
|---|---|---|---|
| 0–10 s | 停在默认光照 | L0 | 这是重建出来的原始外观 |
| 10–25 s | 拖光强滑块，调暗再调亮 | L1 | 整体亮度跟着变，而且是物理的衰减关系，不是简单乘一个数 |
| 25–45 s | **转光束方向**（yaw / pitch） | L2 | **最有说服力的一段**：亮区会跟着光束在肠壁上移动——这是标准 3DGS 做不到的 |
| 45–60 s | **把光源移离相机光心** | L3 | 阴影侧和受光侧出现分离，相机没动但画面完全变了 |
| 60–75 s | 改光源颜色 | L4 | 色彩重新映射，组织基础色不变 |

25–45 s 那一段是整个演示的高潮：**相机完全静止，只有光在动，画面在物理上正确地变化。**
这一句话就说清了 PR-ENDO 相对标准 3DGS 的全部价值。

---

## 怎么放进 PPT

**推荐：嵌入 mp4，设为自动播放 + 循环，不要用超链接。**
超链接会切出 PPT，一旦播放器没装好就冷场。

```
PowerPoint:  插入 → 视频 → 此设备 → 选中 mp4
             播放选项 → 开始：自动 / 循环播放，直到停止 / 播完返回开头
Keynote:     拖进幻灯片 → 检查器 → 影片 → 重复：循环
Google Slides: 只支持 Drive / YouTube，需要先上传；离线演示不推荐
```

**导出参数**：1280×720 或 1920×1080，30 fps，H.264，**不要音轨**（现场是你在讲）。
单段控制在 90 秒、20 MB 以内。

**GIF 的用法不一样**：`../gifs/` 里的 8 个动图是 PR-ENDO P1 预测（上）和 Unity 真值（下）的
逐帧对比，带 PSNR 标注。它们**不是演示工具，是结果证据**——放在讲结果的那几页里自动循环，
让观众自己看"上下两张几乎分不出来"。GIF 在 PPT 里会自动循环，不需要额外设置。

| GIF | 内容 |
|---|---|
| `P1_L0_6pairs.gif` | L0，6 帧 |
| `P1_L2_directions_7pairs.gif` | L2 七个方向各一帧 — **讲重光照时首选这个** |
| `P1_L2_{fwd,yaw+20,yaw-20,pitch+20}_6pairs.gif` | 单个 L2 方向的 6 帧 |
| `P1_L3_f+0.25_6pairs.gif` | L3 光源前移 |
| `P1_L4_red_6pairs.gif` | L4 红光 |

---

## 录屏工具

Linux 下任选其一：

```bash
# 整个屏幕，30 fps，无音轨
ffmpeg -video_size 1920x1080 -framerate 30 -f x11grab -i :0.0 -c:v libx264 -preset fast -crf 20 -an out.mp4

# 只录某个窗口（先用 xwininfo 点一下窗口拿到几何）
xwininfo | grep -E "Absolute|Width|Height"
ffmpeg -video_size <W>x<H> -framerate 30 -f x11grab -i :0.0+<X>,<Y> -c:v libx264 -preset fast -crf 20 -an out.mp4
```

GNOME 自带的 `Ctrl+Alt+Shift+R` 录屏也够用，但输出是 webm，放进 PowerPoint 前要转一下：

```bash
ffmpeg -i in.webm -c:v libx264 -preset fast -crf 20 -an out.mp4
```

剪辑推荐 `losslesscut`（只切不重编码，最快）或 Kdenlive（需要加字幕箭头时）。
