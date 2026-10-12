# 3D Reconstruction and Neural Rendering

基于 University of Pennsylvania CIS 5800 课程 Project A/B 整理的可复现项目。包含传统标定双目重建、2D 坐标神经表示和 NeRF。保留原有算法及权重接口，补充命令行、数据准备、图片/点云导出、小批量训练和验证记录。

**定位：课程实现与工程整理。** 原课程模板、数据加载和框架不是原创研究；目前不宣称新算法、SLAM、动态 world model 或机器人部署。贡献边界见 [PROVENANCE.md](docs/PROVENANCE.md)。

![位置编码图像拟合对比](docs/assets/image_frequency_comparison.png)

当前 L=0/2/6 权重在输入图像上的 PSNR 分别为 19.67/26.61/32.08 dB。其他实际结果及当前 NeRF 的质量限制见 [复现记录](docs/REPRODUCTION.md)。

## 在这台电脑上立即复现

独立环境位于 `.venv`；数据和四个课程权重已准备好。打开 PowerShell：

```powershell
cd 'C:\Users\Lewis\Documents\New project\cis5800-3d-vision'
.\.venv\Scripts\python.exe -m vision5800 stereo
.\.venv\Scripts\python.exe -m vision5800 neural --task image --freq 6 --output outputs/image_eval
.\.venv\Scripts\python.exe -m vision5800 neural --task nerf --output outputs/nerf_eval
```

默认 stereo 使用原图的 1/4 尺寸，便于快速看到点云。CPU NeRF 渲染比图像拟合慢；默认分块降低内存峰值。结果保存在 `outputs/`。本次已执行的命令、指标和限制见 [REPRODUCTION.md](docs/REPRODUCTION.md)。

## 在新电脑安装

建议 Python 3.12。CPU 安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe scripts/prepare_data.py --zip 'C:\path\HWs&Project.zip'
.\.venv\Scripts\python.exe -m pytest -q
```

如已安装适配显卡的 PyTorch，可保留它，再安装项目基础依赖。`requirements-lock.txt` 记录本次 Windows CPU 验证的环境，不是跨平台 CUDA 安装指南。

本地数据与权重不进入 Git。`scripts/prepare_data.py` 从你提供的原始 ZIP 提取所需文件，记录来源和 SHA-256；不会下载、执行压缩包里的脚本。无课程 ZIP 的访问者需要自行取得合法可用的数据并适配相同格式。

## Project A：标定双目重建

```powershell
# 三种匹配代价；每次使用不同输出目录
.\.venv\Scripts\python.exe -m vision5800 stereo --kernel ssd --output outputs/stereo_ssd
.\.venv\Scripts\python.exe -m vision5800 stereo --kernel sad --output outputs/stereo_sad
.\.venv\Scripts\python.exe -m vision5800 stereo --kernel zncc --output outputs/stereo_zncc
# 原 notebook 的九对视图聚合
.\.venv\Scripts\python.exe -m vision5800 stereo --aggregate --kernel sad --output outputs/stereo_aggregate
# 原始尺寸：计算成本显著增加
.\.venv\Scripts\python.exe -m vision5800 stereo --scale 1 --output outputs/stereo_full
```

流程：已知内外参 → 相对位姿 → 双目校正 → SSD/SAD/ZNCC 局部匹配 → 左右一致性 → 深度 → 世界坐标彩色点云。

数据集图像旋转过，校正后的搜索方向是图像的 **v 方向**。默认双目对为 0/3；索引从零开始。保留课程的 0.5–0.6 深度过滤范围及背景过滤，适用于该 Temple 场景，不能直接泛化到任意图片。图像缩小时同步缩放内参。

输出：输入图、视差/深度 PNG、原始数值 NPZ、`reconstruction.ply`、`pointcloud.png`、`metrics.json`。PLY 可用 MeshLab/CloudCompare 打开。PNG 色阶按有效值百分位归一化，跨实验定量比较应使用 NPZ，不能直接比较图像颜色。

## Project B：坐标 MLP 与 NeRF

```powershell
# 已有的 L=0/2/6 图像权重分别评估
.\.venv\Scripts\python.exe -m vision5800 neural --task image --freq 0 --output outputs/image_L0
.\.venv\Scripts\python.exe -m vision5800 neural --task image --freq 2 --output outputs/image_L2
.\.venv\Scripts\python.exe -m vision5800 neural --task image --freq 6 --output outputs/image_L6
# 2D 模型从随机初始化训练
.\.venv\Scripts\python.exe -m vision5800 neural --task image --mode train --freq 6 --iterations 10000 --output outputs/image_train
# NeRF 从随机初始化做低分辨率流程检查
.\.venv\Scripts\python.exe -m vision5800 neural --task nerf --mode train --iterations 5 --scale 0.5 --samples 16 --rays-per-step 128 --output outputs/nerf_smoke
# 更完整训练；有适配的 CUDA 环境时加 --device cuda
.\.venv\Scripts\python.exe -m vision5800 neural --task nerf --mode train --iterations 3000 --samples 64 --output outputs/nerf_train
# 渲染多个数据集中已有相机位姿并导出 GIF
.\.venv\Scripts\python.exe -m vision5800 neural --task nerf --scale 0.5 --render-orbit 12 --output outputs/nerf_sweep
```

NeRF 流程：相机射线 → 确定性均匀深度采样 → 位置/方向编码 → 密度和颜色网络 → 可微体渲染。保留原始采样约定；没有随机分层抖动和 hierarchical sampling。

加载权重使用 `weights_only=True`，并从权重形状推断网络宽度和编码频率。评估输出目标/预测/并排对比 PNG 与 JSON 指标；训练另外保存 CSV 和新权重，不覆盖输入权重。

**评估边界：** 原训练脚本使用图像 101 的 PSNR 选权重，所以现有权重在视图 101 的结果属于验证质量。新训练固定使用 0–89 训练，保留 90–99 验证，训练结束后评估测试视图 100；没有使用测试结果选 checkpoint。新入口采用射线小批量，保持网络与渲染接口，不保证复现原先整幅图训练的数值轨迹。GIF 使用给定相机轨迹，可能包含训练视图，不能把它当作全部未见视角的测试。

## 仓库结构

```text
src/vision5800/
  stereo.py             课程双目核心实现（统计滤波改用 SciPy）
  data.py               课程数据加载接口
  image_field.py        2D 位置编码与 MLP
  nerf.py               原 NeRF 核心实现
  experiments.py        双目实验、聚合和导出
  neural_experiment.py  神经模型评估、小批量训练和 GIF
  artifacts.py          PNG、PLY 和 JSON 输出
  cli.py                统一命令行
scripts/prepare_data.py 本地课程 ZIP 的资产提取
tests/                  几何与渲染关键性质验证
docs/                   来源、复现记录和项目展示建议
data/                   本地输入（Git 忽略）
checkpoints/            本地课程权重（Git 忽略）
outputs/                可重建实验产物（Git 忽略）
```

## 作品集使用

优先展示：双目输入/深度/点云、L=0/2/6 对比、NeRF 目标与预测、相机轨迹 GIF；配上数据集、配置、指标、个人贡献和失败案例。展示前先读 [PORTFOLIO.md](docs/PORTFOLIO.md)。

这份本地仓库尚未发布到 GitHub。保留课程来源说明，公开前确认课程材料及数据的发布条件；当前不添加覆盖第三方材料的开源许可证。
