# TLC-Calib 快速上手

本文档基于本地已下载的数据集（`data/TLC-Calib/KITTI-360` 与 `data/TLC-Calib/FAST-LIVO2`），汇总单场景 / 多场景 / 多卡训练、评估与可视化的常用命令。

## 0. 前置条件

```bash
cd /path/to/TLC-Calib
conda activate tlc_calib
```

数据根目录（下文记为 `DATA_ROOT`）：

```text
data/TLC-Calib/
├── KITTI-360/          # 5 个场景，4 相机
│   ├── straight/
│   ├── small_zigzag/
│   ├── small_rotation/
│   ├── large_zigzag/
│   └── large_rotation/
└── FAST-LIVO2/         # 3 个场景，1 相机
    ├── Building/
    ├── Landmark/
    └── Sculpture/
```

推荐训练参数（论文默认配置）：

```bash
COMMON_ARGS="--eval --from_lidar --use_rig --opt_pose --pose_scheduler --adaptive_voxel"
```

输出目录约定：

```text
outputs/
├── kitti-360/<scene_name>/eval/          # 单场景单次运行
└── fast-livo2/<scene_name>/eval/
```

---

## 1. 数据集概览

### KITTI-360（`--dataset kitti-360`）

本地 `data/TLC-Calib/KITTI-360/` 下共 **5 个场景**（均已解压可用）：

| # | 场景名 (`--scenes`) | 数据路径 | 原始序列 | 原始帧范围 | 本地帧数 | 相机 |
|---|---------------------|----------|----------|------------|----------|------|
| 1 | `straight` | `data/TLC-Calib/KITTI-360/straight` | seq 0009 | 980–1058 | 74 | 4 |
| 2 | `small_zigzag` | `data/TLC-Calib/KITTI-360/small_zigzag` | seq 0010 | 3390–3468 | 63 | 4 |
| 3 | `small_rotation` | `data/TLC-Calib/KITTI-360/small_rotation` | seq 0010 | 98–177 | 80 | 4 |
| 4 | `large_zigzag` | `data/TLC-Calib/KITTI-360/large_zigzag` | seq 0009 | 11601–11680 | 80 | 4 |
| 5 | `large_rotation` | `data/TLC-Calib/KITTI-360/large_rotation` | seq 0009 | 2854–2932 | 79 | 4 |

`--scenes` 一次性指定全部 KITTI-360 场景：

```bash
--scenes straight small_zigzag small_rotation large_zigzag large_rotation
```

### FAST-LIVO2（`--dataset fast-livo2`）

| 场景 | 数据路径 | 原始 bag | 帧范围 | 采样 | 相机数 |
|------|----------|----------|--------|------|--------|
| Building | `data/TLC-Calib/FAST-LIVO2/Building` | CBD_Building_01.bag | 120–357 | 每 3 帧 | 1 |
| Landmark | `data/TLC-Calib/FAST-LIVO2/Landmark` | HKU_Landmark.bag | 45–282 | 每 3 帧 | 1 |
| Sculpture | `data/TLC-Calib/FAST-LIVO2/Sculpture` | HKUST_Sculpture.bag | 50–287 | 每 3 帧 | 1 |

> **注意**：`--scenes` 参数需与文件夹名完全一致。FAST-LIVO2 场景名首字母大写（`Building` / `Landmark` / `Sculpture`），KITTI-360 为小写加下划线（`large_rotation` 等）。

---

## 2. 单场景训练

### 2.1 KITTI-360（5 个场景完整命令）

```bash
# 1/5 straight
python train.py \
  -s data/TLC-Calib/KITTI-360/straight \
  -m outputs/kitti-360/straight/eval \
  $COMMON_ARGS --dataset kitti-360

# 2/5 small_zigzag
python train.py \
  -s data/TLC-Calib/KITTI-360/small_zigzag \
  -m outputs/kitti-360/small_zigzag/eval \
  $COMMON_ARGS --dataset kitti-360

# 3/5 small_rotation
python train.py \
  -s data/TLC-Calib/KITTI-360/small_rotation \
  -m outputs/kitti-360/small_rotation/eval \
  $COMMON_ARGS --dataset kitti-360

# 4/5 large_zigzag
python train.py \
  -s data/TLC-Calib/KITTI-360/large_zigzag \
  -m outputs/kitti-360/large_zigzag/eval \
  $COMMON_ARGS --dataset kitti-360

# 5/5 large_rotation
python train.py \
  -s data/TLC-Calib/KITTI-360/large_rotation \
  -m outputs/kitti-360/large_rotation/eval \
  $COMMON_ARGS --dataset kitti-360
```

批量串行训练全部 KITTI-360 场景：

```bash
for scene in straight small_zigzag small_rotation large_zigzag large_rotation; do
  python train.py \
    -s data/TLC-Calib/KITTI-360/$scene \
    -m outputs/kitti-360/$scene/eval \
    $COMMON_ARGS --dataset kitti-360
done
```

### 2.2 FAST-LIVO2 示例

```bash
# Building
python train.py \
  -s data/TLC-Calib/FAST-LIVO2/Building \
  -m outputs/fast-livo2/Building/eval \
  $COMMON_ARGS --dataset fast-livo2

# Landmark
python train.py \
  -s data/TLC-Calib/FAST-LIVO2/Landmark \
  -m outputs/fast-livo2/Landmark/eval \
  $COMMON_ARGS --dataset fast-livo2

# Sculpture
python train.py \
  -s data/TLC-Calib/FAST-LIVO2/Sculpture \
  -m outputs/fast-livo2/Sculpture/eval \
  $COMMON_ARGS --dataset fast-livo2
```

### 2.3 训练时实时可视化（Web Viewer）

在训练命令末尾追加：

```bash
--viewer --port 8080 --viewer_camera_step 3
```

浏览器访问 `http://127.0.0.1:8080` 查看标定过程。长序列可增大 `--viewer_camera_step` 降低刷新负载。

### 2.4 可选：NVS 精修阶段

在 `$COMMON_ARGS` 后追加 `--refine`，会在标定完成后额外运行 NVS 精修（默认 +10000 iter）。

---

## 3. 单场景评估

训练完成后，对单个输出目录运行：

```bash
OUT=outputs/kitti-360/large_rotation/eval   # 替换为实际输出路径

# 标定位姿误差（Rot / Trans）
python metrics_pose.py -m $OUT
# 结果: $OUT/rig_results.json

# NVS 质量（PSNR / SSIM / LPIPS）
python metrics_nvs.py -m $OUT --model_iter 30000
# 结果: $OUT/nvs_results.json
#       $OUT/nvs_eval/ours_30000/
```

KITTI-360 全部 5 场景批量评估：

```bash
for scene in straight small_zigzag small_rotation large_zigzag large_rotation; do
  OUT=outputs/kitti-360/$scene/eval
  python metrics_pose.py -m $OUT
  python metrics_nvs.py -m $OUT --model_iter 30000
done
```

各场景输出路径对照：

| 场景 | 评估目录 `$OUT` |
|------|-----------------|
| straight | `outputs/kitti-360/straight/eval` |
| small_zigzag | `outputs/kitti-360/small_zigzag/eval` |
| small_rotation | `outputs/kitti-360/small_rotation/eval` |
| large_zigzag | `outputs/kitti-360/large_zigzag/eval` |
| large_rotation | `outputs/kitti-360/large_rotation/eval` |

---

## 4. 单场景可视化（渲染）

`render.py` 直接渲染已训练的标定模型，用于检查标定效果（不重新训练 NVS，也不计算 PSNR/SSIM/LPIPS）：

```bash
OUT=outputs/kitti-360/large_rotation/eval

python render.py -m $OUT --iteration 30000
```

输出目录：

```text
$OUT/train/ours_30000/   # 训练视角渲染、GT、误差图
$OUT/test/ours_30000/    # 测试视角渲染、GT、误差图
```

仅渲染测试集：

```bash
python render.py -m $OUT --iteration 30000 --skip_train
```

KITTI-360 全部 5 场景批量渲染：

```bash
for scene in straight small_zigzag small_rotation large_zigzag large_rotation; do
  python render.py -m outputs/kitti-360/$scene/eval --iteration 30000
done
```

---

## 5. 多场景训练

### 5.1 串行：逐个场景训练（单卡）

使用 `full_eval.py` 的 `--skip_nvs` 可只做训练 + 位姿评估，跳过耗时的 NVS 评估：

```bash
# KITTI-360 全部 5 场景
python full_eval.py \
  --data_path data/TLC-Calib \
  --output_path outputs \
  --datasets kitti-360 \
  --skip_nvs \
  $COMMON_ARGS

# FAST-LIVO2 全部 3 场景
python full_eval.py \
  --data_path data/TLC-Calib \
  --output_path outputs \
  --datasets fast-livo2 \
  --skip_nvs \
  $COMMON_ARGS

# 两个数据集一起跑
python full_eval.py \
  --data_path data/TLC-Calib \
  --output_path outputs \
  --datasets kitti-360 fast-livo2 \
  --skip_nvs \
  $COMMON_ARGS
```

指定部分场景（以下为 KITTI-360 全部 5 个场景名）：

```bash
python full_eval.py \
  --data_path data/TLC-Calib \
  --output_path outputs \
  --datasets kitti-360 \
  --scenes straight small_zigzag small_rotation large_zigzag large_rotation \
  --skip_nvs \
  $COMMON_ARGS
```

### 5.2 并行：多场景多卡训练（仅训练，不评估）

通过 `train.py --parallel`，每个场景占用一张 GPU 并行训练。**并行模式下需显式传入 `$COMMON_ARGS`**（不会自动补全默认参数）：

```bash
# KITTI-360 五场景 × 4 卡
python train.py --parallel \
  -d data/TLC-Calib \
  -m outputs \
  --datasets kitti-360 \
  --gpus 0 1 2 3 \
  $COMMON_ARGS

# FAST-LIVO2 三场景 × 3 卡
python train.py --parallel \
  -d data/TLC-Calib \
  -m outputs \
  --datasets fast-livo2 \
  --gpus 0 1 2 \
  $COMMON_ARGS

# 混合数据集 + 指定 GPU
python train.py --parallel \
  -d data/TLC-Calib \
  -m outputs \
  --datasets kitti-360 fast-livo2 \
  --gpus 0 1 2 3 \
  $COMMON_ARGS

# 使用全部可见 GPU
python train.py --parallel \
  -d data/TLC-Calib \
  -m outputs \
  --datasets kitti-360 \
  --gpus all \
  $COMMON_ARGS
```

输出路径：`outputs/<dataset>/<scene>/eval/`。

---

## 6. 多场景评估（训练 + 位姿 + NVS + 汇总）

### 6.1 串行全量评估

```bash
# KITTI-360 全场景（训练 + pose + NVS + 场景级汇总）
python full_eval.py \
  --data_path data/TLC-Calib \
  --output_path outputs \
  --datasets kitti-360 \
  --repeat 1 \
  $COMMON_ARGS

# FAST-LIVO2 全场景
python full_eval.py \
  --data_path data/TLC-Calib \
  --output_path outputs \
  --datasets fast-livo2 \
  --repeat 1 \
  $COMMON_ARGS

# 两个数据集一起评估
python full_eval.py \
  --data_path data/TLC-Calib \
  --output_path outputs \
  --datasets kitti-360 fast-livo2 \
  --repeat 1 \
  $COMMON_ARGS
```

### 6.2 并行：多场景多卡全量评估

每个 job（场景 × repeat）分配到一张 GPU，自动完成 train → metrics_pose → metrics_nvs → 汇总：

```bash
# KITTI-360，4 卡并行
python full_eval.py --parallel \
  --data_path data/TLC-Calib \
  --output_path outputs \
  --datasets kitti-360 \
  --gpus 0 1 2 3 \
  --repeat 1 \
  $COMMON_ARGS

# FAST-LIVO2，3 卡并行
python full_eval.py --parallel \
  --data_path data/TLC-Calib \
  --output_path outputs \
  --datasets fast-livo2 \
  --gpus 0 1 2 \
  $COMMON_ARGS

# 混合数据集，使用全部 GPU
python full_eval.py --parallel \
  --data_path data/TLC-Calib \
  --output_path outputs \
  --datasets kitti-360 fast-livo2 \
  --gpus all \
  $COMMON_ARGS
```

> 指定 `--gpus` 时会自动开启 `--parallel`，也可显式加 `-p`。

### 6.3 多次重复实验

```bash
python full_eval.py --parallel \
  --data_path data/TLC-Calib \
  --output_path outputs \
  --datasets kitti-360 \
  --scenes large_rotation \
  --gpus 0 1 \
  --repeat 3 \
  $COMMON_ARGS
```

输出为 `eval/`（repeat=1）或 `eval_01/`、`eval_02/`、`eval_03/`（repeat>1），并自动生成场景级平均结果。

### 6.4 跳过 NVS 加速

```bash
python full_eval.py --parallel \
  --data_path data/TLC-Calib \
  --output_path outputs \
  --datasets kitti-360 fast-livo2 \
  --gpus all \
  --skip_nvs \
  $COMMON_ARGS
```

---

## 7. 汇总已有结果（不重新训练）

对已完成的输出目录做聚合，无需重跑训练：

```bash
# 单场景汇总（生成 exp_rig_results.json / exp_nvs_results.json / train_results.json）
python full_eval.py -a outputs/kitti-360/large_rotation

# 单数据集汇总（生成 full_eval_results.json）
python full_eval.py -a outputs/kitti-360

# 全部输出根目录汇总
python full_eval.py -a outputs
```

---

## 8. 输出文件说明

单次运行（`eval/`）典型输出：

| 文件 / 目录 | 说明 |
|-------------|------|
| `config.yml` | 训练与数据集配置 |
| `point_cloud/iteration_30000/` | 优化后的 Gaussian 与 `cams_to_lidar.txt` |
| `train_info.json` | 训练耗时与显存统计 |
| `outputs.log` | 训练日志 |
| `rig_results.json` | 各相机 Rot/Trans 标定误差 |
| `nvs_results.json` | NVS PSNR/SSIM/LPIPS |
| `train/ours_30000/` | 渲染结果（若执行了 render 或 eval 流程） |

场景级汇总（`full_eval.py` 自动生成）：

| 文件 | 说明 |
|------|------|
| `exp_rig_results.json` | 多次 run 平均位姿误差 |
| `exp_nvs_results.json` | 多次 run 平均 NVS 指标 |
| `train_results.json` | 平均训练时间与显存 |
| `full_eval_results.json` | 数据集级全场景汇总 |

---

## 9. 常用命令速查

```bash
# ── KITTI-360 五场景（train / eval / render）────────
for scene in straight small_zigzag small_rotation large_zigzag large_rotation; do
  python train.py -s data/TLC-Calib/KITTI-360/$scene -m outputs/kitti-360/$scene/eval \
    --eval --from_lidar --use_rig --opt_pose --pose_scheduler --adaptive_voxel --dataset kitti-360
  python metrics_pose.py -m outputs/kitti-360/$scene/eval
  python metrics_nvs.py -m outputs/kitti-360/$scene/eval --model_iter 30000
  python render.py -m outputs/kitti-360/$scene/eval --iteration 30000
done

# ── 多场景多卡训练 ──────────────────────────────────
python train.py --parallel -d data/TLC-Calib -m outputs \
  --datasets kitti-360 fast-livo2 --gpus all \
  --eval --from_lidar --use_rig --opt_pose --pose_scheduler --adaptive_voxel

# ── 多场景多卡训练+评估+汇总 ────────────────────────
python full_eval.py --parallel -d data/TLC-Calib -o outputs \
  --datasets kitti-360 fast-livo2 --gpus all --repeat 1 \
  --eval --from_lidar --use_rig --opt_pose --pose_scheduler --adaptive_voxel

# ── 汇总已有结果 ────────────────────────────────────
python full_eval.py -a outputs
```

---

## 10. 参考

- 项目 README：[README.md](./README.md)
- 数据格式说明：[data/TLC-Calib/README.md](data/TLC-Calib/README.md)
- 场景配置表：[data/TLC-Calib/CONFIG.md](data/TLC-Calib/CONFIG.md)
