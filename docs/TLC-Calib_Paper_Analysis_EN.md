# TLC-Calib Paper In-Depth Analysis Report

> **Paper**: Targetless LiDAR-Camera Calibration with Neural Gaussian Splatting  
> **Authors**: Haebeom Jung, Namtae Kim, Jungwoo Kim, Jaesik Park  
> **Affiliations**: Seoul National University, Yonsei University  
> **Publication**: IEEE Robotics and Automation Letters (RA-L) 2026, ICRA 2026  
> **Repository**: https://github.com/SNU-VGILab/TLC-Calib

---

## Table of Contents

- [1. Research Background and Motivation](#1-research-background-and-motivation)
- [2. Method Overview](#2-method-overview)
- [3. Core Technical Details](#3-core-technical-details)
  - [3.1 Pose-Differentiable Rasterization](#31-pose-differentiable-rasterization)
  - [3.2 Anchor Gaussians](#32-anchor-gaussians)
  - [3.3 Auxiliary Gaussians](#33-auxiliary-gaussians)
  - [3.4 Camera Rig Optimization Strategy](#34-camera-rig-optimization-strategy)
  - [3.5 Adaptive Voxel Control (AVC)](#35-adaptive-voxel-control-adaptive-voxel-control-avc)
  - [3.6 Loss Functions](#36-loss-functions)
  - [3.7 Pose Learning Rate Scheduling](#37-pose-learning-rate-scheduling)
  - [3.8 From-LiDAR Initialization](#38-from-lidar-initialization)
- [4. Deep Analysis of Pose Gradient Propagation in the CUDA Rasterizer](#4-deep-analysis-of-pose-gradient-propagation-in-the-cuda-rasterizer)
  - [4.1 Custom Rasterizer Architecture](#41-custom-rasterizer-architecture)
  - [4.2 Forward Pass](#42-forward-pass)
  - [4.3 Backward Pass — Four Gradient Paths](#43-backward-pass--four-gradient-paths)
  - [4.4 Gradient Aggregation and Python Binding](#44-gradient-aggregation-and-python-binding)
  - [4.5 Lie Algebra Mathematical Foundations (CUDA Implementation)](#45-lie-algebra-mathematical-foundations-cuda-implementation)
  - [4.6 Complete Gradient Flow Diagram](#46-complete-gradient-flow-diagram)
- [5. How the Refine Stage Works](#5-how-the-refine-stage-works)
- [6. Code Differences vs. Scaffold-GS / MonoGS](#6-code-differences-vs-scaffold-gs--monogs)
- [7. Data Preprocessing and Custom Dataset Integration](#7-data-preprocessing-and-custom-dataset-integration)
- [8. NVS Evaluation Module Workflow](#8-nvs-evaluation-module-workflow)
- [9. Web Viewer Real-Time Visualization Module](#9-web-viewer-real-time-visualization-module)
- [10. nvs_eval/ Submodule: Standard 3DGS Evaluation Pipeline](#10-nvs_eval-submodule-standard-3dgs-evaluation-pipeline)
- [11. full_eval.py Batch Evaluation System](#11-full_evalpy-batch-evaluation-system)
- [12. Consistency Between Paper Experiments and Code Hyperparameters](#12-consistency-between-paper-experiments-and-code-hyperparameters)
- [13. Supplementary Analysis of Core Contributions](#13-supplementary-analysis-of-core-contributions)
  - [13.1 Loss Landscape Smoothing Mechanism](#131-loss-landscape-smoothing-mechanism)
  - [13.2 Time Offset Modeling](#132-time-offset-modeling)
  - [13.3 Warmup and Convergence Monitoring](#133-warmup-and-convergence-monitoring)
  - [13.4 AdamW Weight Decay Strategy](#134-adamw-weight-decay-strategy)
  - [13.5 Noise Injection and Robustness Testing](#135-noise-injection-and-robustness-testing)
  - [13.6 Voxel Prefiltering Acceleration](#136-voxel-prefiltering-acceleration)
  - [13.7 Extrinsic Generalization Capability](#137-extrinsic-generalization-capability)
  - [13.8 Depth Rendering and Gradient Propagation](#138-depth-rendering-and-gradient-propagation)
- [14. Deep Analysis of the CUDA Forward Rendering Pipeline](#14-deep-analysis-of-the-cuda-forward-rendering-pipeline)
  - [14.1 Tile-Based Rasterization Overall Architecture](#141-tile-based-rasterization-overall-architecture)
  - [14.2 preprocessCUDA: Per-Gaussian Preprocessing Kernel](#142-preprocesscuda-per-gaussian-preprocessing-kernel)
  - [14.3 renderCUDA: Per-Pixel Alpha-Blending Kernel](#143-rendercuda-per-pixel-alpha-blending-kernel)
  - [14.4 Sorting and Per-Tile Scheduling](#144-sorting-and-per-tile-scheduling)
  - [14.5 filter_preprocessCUDA: Voxel Prefiltering Dedicated Kernel](#145-filter_preprocesscuda-voxel-prefiltering-dedicated-kernel)
  - [14.6 Symmetric Design of Forward and Backward Passes](#146-symmetric-design-of-forward-and-backward-passes)
- [15. render.py Visualization Rendering Script Analysis](#15-renderpy-visualization-rendering-script-analysis)
  - [15.1 Script Purpose and Role](#151-script-purpose-and-role)
  - [15.2 render_set Function Details](#152-render_set-function-details)
  - [15.3 Output Directory Structure](#153-output-directory-structure)
  - [15.4 Relationship with train.py and metrics_nvs.py](#154-relationship-with-trainpy-and-metrics_nvspy)
- [16. TLC-Calib vs. Latest 3DGS Calibration Methods](#16-tlc-calib-vs-latest-3dgs-calibration-methods)
  - [16.1 Method Lineage Diagram](#161-method-lineage-diagram)
  - [16.2 Per-Method Deep Comparison](#162-per-method-deep-comparison)
  - [16.3 Quantitative Performance Comparison](#163-quantitative-performance-comparison)
  - [16.4 Technical Approach Comparison Table](#164-technical-approach-comparison-table)
  - [16.5 TLC-Calib Differentiating Advantages](#165-tlc-calib-differentiating-advantages)
- [17. Deep Analysis of CUDA Backward Rendering Kernels](#17-deep-analysis-of-cuda-backward-rendering-kernels)
  - [17.1 Reverse Traversal Mechanism in Backward renderCUDA](#171-reverse-traversal-mechanism-in-backward-rendercuda)
  - [17.2 Per-Pixel Gradient Decomposition](#172-per-pixel-gradient-decomposition)
  - [17.3 Warp-Level Reduce Sum Optimization](#173-warp-level-reduce-sum-optimization)
  - [17.4 Pose Gradient Injection in computeCov2DCUDA](#174-pose-gradient-injection-in-computecov2dcuda)
  - [17.5 Backward preprocessCUDA: Four-Path Convergence of Pose Gradients](#175-backward-preprocesscuda-four-path-convergence-of-pose-gradients)
- [18. Python Binding Layer autograd Function Analysis](#18-python-binding-layer-autograd-function-analysis)
  - [18.1 _RasterizeGaussians forward/backward Mechanism](#181-_rasterizegaussians-forwardbackward-mechanism)
  - [18.2 Pose Gradient Aggregation and Splitting](#182-pose-gradient-aggregation-and-splitting)
  - [18.3 GaussianRasterizer Complete Call Chain](#183-gaussianrasterizer-complete-call-chain)
  - [18.4 Lightweight Path of visible_filter](#184-lightweight-path-of-visible_filter)
- [19. TLC-Calib vs. NeRF-Based Calibration Methods](#19-tlc-calib-vs-nerf-based-calibration-methods)
  - [19.1 Overview of NeRF-Based Calibration Methods](#191-overview-of-nerf-based-calibration-methods)
  - [19.2 Per-Method Comparison: INF / MOISST / SOAC](#192-per-method-comparison-inf--moisst--soac)
  - [19.3 NeRF vs 3DGS: Paradigm-Level Differences](#193-nerf-vs-3dgs-paradigm-level-differences)
  - [19.4 Cross-Method Comparison Matrix](#194-cross-method-comparison-matrix)
- [20. GPU Memory Management Architecture (rasterizer_impl)](#20-gpu-memory-management-architecture-rasterizer_impl)
  - [20.1 Chunk-Based Memory Allocation Pattern](#201-chunk-based-memory-allocation-pattern)
  - [20.2 Three Major State Buffers](#202-three-major-state-buffers)
  - [20.3 Buffer Reuse Between Forward and Backward](#203-buffer-reuse-between-forward-and-backward)
- [21. cameras.py Camera Model and Projection Matrix](#21-cameraspy-camera-model-and-projection-matrix)
  - [21.1 Camera Class State Management](#211-camera-class-state-management)
  - [21.2 Projection Matrix Calculation](#212-projection-matrix-calculation)
  - [21.3 Relationship Between Three Projection Matrices](#213-relationship-between-three-projection-matrices)
  - [21.4 MiniCam: Lightweight Camera for Web Viewer](#214-minicam-lightweight-camera-for-web-viewer)
- [22. train.py vs nvs_eval/train.py Comparison](#22-trainpy-vs-nvs_evaltrainpy-comparison)
  - [22.1 Design Purpose Differences](#221-design-purpose-differences)
  - [22.2 Module-by-Module Comparison](#222-module-by-module-comparison)
  - [22.3 Independent Pipeline Design for NVS Evaluation](#223-independent-pipeline-design-for-nvs-evaluation)
- [23. math.h SE(3)/SO(3) CUDA Implementation](#23-mathh-se3so3-cuda-implementation)
  - [23.1 Matrix Type Hierarchy](#231-matrix-type-hierarchy)
  - [23.2 SO(3) Rotation Group Implementation](#232-so3-rotation-group-implementation)
  - [23.3 SE(3) Rigid Transform Group Implementation](#233-se3-rigid-transform-group-implementation)
  - [23.4 CUDA vs Python Dual Implementation Comparison](#234-cuda-vs-python-dual-implementation-comparison)
  - [23.5 Key Role of SE(3) in Gradient Propagation](#235-key-role-of-se3-in-gradient-propagation)
- [24. dataset_readers.py Dataset Format Parsing Differences](#24-dataset_readerspy-dataset-format-parsing-differences)
  - [24.1 Unified Data Directory Structure](#241-unified-data-directory-structure)
  - [24.2 From-LiDAR vs From-Blueprint Initialization](#242-from-lidar-vs-from-blueprint-initialization)
  - [24.3 get_c2l(): Dataset Prior Extrinsics](#243-get_c2l-dataset-prior-extrinsics)
  - [24.4 Specific Differences Across Three Datasets](#244-specific-differences-across-three-datasets)
  - [24.5 Time Offset Handling](#245-time-offset-handling)
  - [24.6 Adaptive Voxel Control (AVC) Voxel Size Computation](#246-adaptive-voxel-control-avc-voxel-size-computation)
- [25. gaussian_model.py Neural Gaussian Complete Initialization and Optimization](#25-gaussian_modelpy-neural-gaussian-complete-initialization-and-optimization)
  - [25.1 Neural Gaussian Parameter Composition](#251-neural-gaussian-parameter-composition)
  - [25.2 Point Cloud Initialization (create_from_pcd)](#252-point-cloud-initialization-create_from_pcd)
  - [25.3 Optimizer Configuration and LR Scheduling](#253-optimizer-configuration-and-lr-scheduling)
  - [25.4 Pose State Management](#254-pose-state-management)
  - [25.5 Anchor Growing and Pruning (AVC)](#255-anchor-growing-and-pruning-avc)
- [26. loss_utils.py Loss Functions and Gradient Contribution Deep Analysis](#26-loss_utilspy-loss-functions-and-gradient-contribution-deep-analysis)
  - [26.1 Total Loss Function Composition](#261-total-loss-function-composition)
  - [26.2 Photometric Loss L_photo (L1)](#262-photometric-loss-l_photo-l1)
  - [26.3 Structural Similarity Loss L_ssim (D-SSIM)](#263-structural-similarity-loss-l_ssim-d-ssim)
  - [26.4 Scale Regularization L_scale](#264-scale-regularization-l_scale)
  - [26.5 Gradient Flow Comparison of Three Loss Terms](#265-gradient-flow-comparison-of-three-loss-terms)
- [27. Real-Vehicle Deployment and Cross-Vehicle Generalization Audit](#27-real-vehicle-deployment-and-cross-vehicle-generalization-audit)
  - [27.1 Current Framework Design Positioning](#271-current-framework-design-positioning)
  - [27.2 Real-Vehicle Deployment Feasibility Analysis](#272-real-vehicle-deployment-feasibility-analysis)
  - [27.3 Cross-Vehicle Generalization Capability Assessment](#273-cross-vehicle-generalization-capability-assessment)
  - [27.4 Core Bottlenecks and Improvement Recommendations](#274-core-bottlenecks-and-improvement-recommendations)
  - [27.5 Recommended Real-Vehicle Deployment Schemes](#275-recommended-real-vehicle-deployment-schemes)
  - [27.6 Summary Assessment](#276-summary-assessment)
- [28. Experimental Results and Analysis](#28-experimental-results-and-analysis)
  - [28.1 KITTI-360 Calibration Accuracy](#281-kitti-360-calibration-accuracy)
  - [28.2 NVS Performance](#282-nvs-performance)
  - [28.3 Training Efficiency](#283-training-efficiency)
  - [28.4 Key Ablation Findings](#284-key-ablation-findings)
- [29. Complete Code Architecture and Data Flow](#29-complete-code-architecture-and-data-flow)
- [30. Summary of Key Design Insights](#30-summary-of-key-design-insights)

---

## 1. Research Background and Motivation

### 1.1 Core Problem

In LiDAR-Camera multi-sensor systems, **extrinsic** calibration between sensors is critical. Extrinsics describe the relative pose relationship between sensors (rotation R + translation t), i.e., the camera-to-lidar 4×4 homogeneous transformation matrix T_c2l. In practice, two major challenges arise:

1. **Target-based methods** require physical targets such as checkerboards and spherical reflectors, incurring high deployment costs and poor accuracy at long distances
2. **Extrinsic drift**: Even after careful calibration, mechanical vibration, thermal expansion, and physical impacts can cause extrinsics to shift over time, requiring periodic re-calibration

### 1.2 Limitations of Existing Targetless Methods

| Method Category | Representative Work | Main Deficiencies |
|----------|----------|----------|
| Edge alignment | Levinson et al. [10] | Limited by feature extraction quality |
| Deep learning | RegNet [13], LCCNet [14] | Requires large-scale labeled data; poor generalization |
| NeRF-based | INF [15], MOISST [16], UniCal [17] | Implicit volumetric representation; extremely high computational cost (>4h) |
| 3DGS-based | 3DGS-Calib [23] | Relies on hash-grid encoding; sensitive to hyperparameters |
| 2DGS-based | RobustCalib [24] | Relies on surface normal estimation; degrades under sparse LiDAR |

**The most critical common deficiency**: Both 3DGS-Calib [23] and RobustCalib [24] **model only LiDAR-observable regions**, fixing Gaussians on LiDAR points. This leaves LiDAR blind zones such as sky and building rooftops without gradient backpropagation, discarding a large amount of valuable photometric supervision signal.

### 1.3 Core Paper Contributions

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          TLC-Calib Four Core Contributions                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ① Neural Gaussian Splatting Framework                                          │
│     ├── Anchor Gaussians: Fixed on LiDAR points, providing global scale constraint│
│     └── Auxiliary Gaussians: MLP-generated dynamically, filling LiDAR blind zones + smoothing loss landscape│
│                                                                                 │
│  ② Pose-Differentiable CUDA Rasterizer                                         │
│     ├── Analytic SE(3) gradients implemented in CUDA kernels                    │
│     └── Four gradient paths: 2D mean + 2D covariance + depth + SH color (retained but inactive)│
│                                                                                 │
│  ③ Camera Rig Optimization                                                      │
│     ├── Shared extrinsics: All frames from the same physical camera share one set│
│     ├── SE(3) incremental accumulation: Geometrically consistent pose updates   │
│     └── Warmup + cosine annealing: Stable convergence strategy                  │
│                                                                                 │
│  ④ Adaptive Voxel Control (AVC)                                                 │
│     ├── Automatically determines anchor density based on LiDAR trajectory length│
│     └── Binary search: V_target = β × D_traj                                    │
│                                                                                 │
│  Experimental results:                                                          │
│  • 100% success rate (vs. INF 74.5%)                                            │
│  • Rotation error 0.13° (vs. INF 0.41°)                                         │
│  • Translation error 8.86cm (vs. INF 32.6cm)                                    │
│  • Training time ~11min (vs. INF >4h)                                         │
│  • GPU memory <8GB                                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Method Overview

The core idea of TLC-Calib: **Build a fully differentiable scene representation based on Neural Gaussian Splatting, jointly optimizing scene representation and sensor extrinsics.** The system uses LiDAR as the reference sensor; its coordinate frame serves as the global reference.

### Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TLC-Calib Pipeline                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input Layer                                                                  │
│  ┌───────────┐  ┌───────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ LiDAR     │  │ LiDAR Poses   │  │ Multi-camera │  │ Camera Intrinsics│   │
│  │ Aggregated│  │ T^L_{1:T}    │  │ Images       │  │ K                │   │
│  │ Point Cloud│  │               │  │ I_{c,t}      │  │ (fx,fy,cx,cy)   │   │
│  └─────┬─────┘  └──────┬────────┘  └──────┬───────┘  └──────┬───────────┘   │
│        │               │                   │                 │               │
│        ▼               ▼                   │                 │               │
│  ┌─────────────────────────┐               │                 │               │
│  │ AVC: Adaptive Voxel     │               │                 │               │
│  │ Downsampling            │               │                 │               │
│  │ V_target = β × D_traj   │               │                 │               │
│  └──────────┬──────────────┘               │                 │               │
│             │                              │                 │               │
│             ▼                              │                 │               │
│  ┌──────────────────────┐                  │                 │               │
│  │ Anchor Gaussians     │◄── Position frozen (lr=0)           │               │
│  │ {v_i, f_i, s_i}      │                  │                 │               │
│  └──────────┬───────────┘                  │                 │               │
│             │ MLP                           │                 │               │
│             ▼                              │                 │               │
│  ┌──────────────────────┐                  │                 │               │
│  │ Auxiliary Gaussians   │                  │                 │               │
│  │ {m_{i,k}, c, σ, s, r}│                  │                 │               │
│  └──────────┬───────────┘                  │                 │               │
│             │                              ▼                 ▼               │
│  ┌────────────────────────────────────────────────────────────┐               │
│  │        Pose-Differentiable Rasterizer (CUDA)               │               │
│  │  Input: 3D Gaussians + T_c(θ,ρ) + K                        │               │
│  │  Output: I'_{c,t} (rendered image) + depth_map             │               │
│  └──────────┬─────────────────────────────────────────────────┘               │
│             │                                                                │
│             ▼                                                                │
│  ┌──────────────────────────────────────────┐                                │
│  │ L_total = (1-λ)·L1 + λ·D-SSIM + L_scale │                                │
│  └──────────┬───────────────────────────────┘                                │
│             │ .backward()                                                    │
│             ▼                                                                │
│  ┌──────────────────────────────────────────────────┐                        │
│  │ Dual-channel gradient flow:                       │                        │
│  │   ① Scene Optimizer (AdamW): MLP, feat, offset   │                        │
│  │   ② Calib Optimizer (Adam): θ_rot, ρ_trans       │                        │
│  │      └─ SE3_exp incremental accumulation to accum_P│                      │
│  └──────────────────────────────────────────────────┘                        │
│                                                                             │
│  Output: camera-to-lidar extrinsics T^e_c  (cams_to_lidar.txt)              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Training Stage Timeline

```
     Iteration:  0        5×N_img        15K         30K         40K
                 │          │              │           │           │
    ─────────────┼──────────┼──────────────┼───────────┼───────────┤
                 │          │              │           │           │
    Warmup ──────┤          │              │           │           │
    (pose frozen)│          │              │           │           │
                 │          │              │           │           │
    Pose Opt ────────────────┤              │           │           │
    (cosine lr)               │              │           │           │
                             │              │           │           │
    Densification ───────────┤              │           │           │
    (prune only, no growth)  │              │           │           │
                             │              │           │           │
    Weight Decay ────────────────────────────┤           │           │
    (AdamW 1e-2)                             │(set to 0) │           │
                                             │           │           │
    Scene Decay ─────────────────────────────┤           │           │
                                              │          │           │
    ──────────── Main Training (30K) ─────────┤          │           │
                                              │          │           │
    Pose Frozen ─────────────────────────────────────────────┤           │
    Refine Densification ────────────────────────────────┤           │
    (growth + prune)                                       │           │
                                                         │           │
    ──── Refine (10K, optional) ──────────────────────────┤           │
```

---

## 3. Core Technical Details

### 3.1 Pose-Differentiable Rasterization

**Paper Equations (1)-(3)**: Propagate photometric loss gradients to camera pose via the chain rule.

Rendered color (standard alpha compositing):

```
C(u) = Σ_i c_i · α_i · G^{2D}_i(u) · Π_{j<i} (1 - α_j · G^{2D}_j(u))
```

Chain rule for pose gradients (**Paper Equation (2)**):

```
∂L/∂T_c = Σ_i ( ∂L/∂μ^{2D}_i · ∂μ^{2D}_i/∂μ^c_i · ∂μ^c_i/∂T_c
              + ∂L/∂Σ^{2D}_i · ∂Σ^{2D}_i/∂T_c
              + ∂L/∂c_i · ∂c_i/∂T_c )
```

Pose updates are performed on the SE(3) Lie group (**Paper Equation (3)**):

```
T_c ← exp(-η · ∂L/∂τ) · T_c,   τ = [ρ, θ] ∈ se(3)
```

**Code correspondence**: The repository uses a custom CUDA rasterizer `diff-gaussian-rasterization-w-pose` (suffix `-w-pose` distinguishes it from the standard version). In `gaussian_renderer/__init__.py`, pose increment parameters `theta` (rotation) and `rho` (translation) are passed for analytic pose gradient computation at the CUDA level.

The Python implementation of the SE(3) exponential map is in `utils/pose_utils.py::SE3_exp()`, where `SO3_exp` uses the Rodrigues formula and `SO3_left_jacobian` computes the left Jacobian matrix.

### 3.2 Anchor Gaussians

**Paper Section III-C.1**: Reliable points are selected from the aggregated LiDAR point cloud as anchor Gaussians. Their **positions remain frozen during training**, providing global scale constraints to prevent calibration drift.

Each anchor carries the following parameters:

| Parameter | Dimensions | Learnable | Description |
|------|------|-----------|------|
| `_anchor` | [N, 3] | **Frozen** (lr=0) | 3D position, taken directly from LiDAR point coordinates |
| `_anchor_feat` | [N, 32] | Learnable | Local geometric feature vector |
| `_offset` | [N, K, 3] | Learnable | Offsets for K auxiliary Gaussians |
| `_scaling` | [N, 6] | Learnable | Anisotropic scale (3 anchor + 3 offset) |
| `_rotation` | [N, 4] | Not learnable | Quaternion rotation |
| `_opacity` | [N, 1] | Not learnable | Opacity |

**Freezing implementation**: In `arguments/__init__.py`, `position_lr_init = 0.0` and `position_lr_final = 0.0`, so anchor positions are never updated by gradients.

```
Key code (arguments/__init__.py):
    self.position_lr_init = 0.0   ← anchor position frozen
    self.position_lr_final = 0.0  ← always zero

Key code (gaussian_model.py::training_setup):
    {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"}
    ↑ 0.0 × any_scale = 0.0, no gradient update
```

**Floater pruning**: Low-opacity anchors are treated as "floaters" and pruned in `gaussian_model.py::adjust_anchor()` based on the condition `opacity_accum < min_opacity * anchor_demon`.

### 3.3 Auxiliary Gaussians

**Paper Section III-C.2**: The core innovation distinguishing TLC-Calib from prior methods. Each anchor generates K auxiliary Gaussians via lightweight MLPs.

Auxiliary Gaussian center position: `m_{i,k} = v_i + δ_{i,k} ⊙ s_i`

Architecture of the three MLPs (all two-layer + activation):

```
┌──────────────────────────────────────────────────────────────────────┐
│                      Auxiliary Gaussian MLP Architecture              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input: [anchor_feat(32), view_direction(3)] = 35 dimensions         │
│                    ↓                                                 │
│  ┌────────────────────────────────────────────┐                      │
│  │ mlp_opacity:                                │                      │
│  │   Linear(35, 32) → ReLU → Linear(32, K)    │                      │
│  │   → Tanh (output range [-1, 1])              │                      │
│  │   Physical meaning: opacity per auxiliary Gaussian│                 │
│  └────────────────────────────────────────────┘                      │
│                                                                      │
│  ┌────────────────────────────────────────────┐                      │
│  │ mlp_cov:                                    │                      │
│  │   Linear(35, 32) → ReLU → Linear(32, 7K)   │                      │
│  │   No activation function                      │                      │
│  │   Physical meaning: scale(3) + quaternion(4) = 7 per auxiliary │  │
│  └────────────────────────────────────────────┘                      │
│                                                                      │
│  ┌────────────────────────────────────────────┐                      │
│  │ mlp_color:                                  │                      │
│  │   Linear(35, 32) → ReLU → Linear(32, 3K)   │                      │
│  │   → Sigmoid (output range [0, 1])            │                      │
│  │   Physical meaning: RGB color               │                      │
│  └────────────────────────────────────────────┘                      │
│                                                                      │
│  Post-processing (gaussian_renderer/__init__.py):                      │
│    scaling = anchor_scale[:,3:] × sigmoid(mlp_output[:,:3])          │
│    rotation = F.normalize(mlp_output[:,3:7])                         │
│    xyz = anchor_pos + offset × anchor_scale[:,:3]                    │
└──────────────────────────────────────────────────────────────────────┘
```

**Four roles of auxiliary Gaussians**:
1. **Fill LiDAR blind zones** (sky, building rooftops), preserving full-image photometric supervision
2. **View-dependent modeling** (MLP input includes viewing direction), improving rendering quality
3. **Smooth the loss landscape**, avoiding pose trapped in local minima (see [Section 13.1](#131-loss-landscape-smoothing-mechanism))
4. **Opacity mask** automatically filters invalid Gaussians (`neural_opacity > 0`), dynamically adjusting effective point count

### 3.4 Camera Rig Optimization Strategy

**Paper Section III-D**: Per-image sequential camera rig optimization.

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Camera Rig Optimization Data Flow                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  When use_rig=True:                                                  │
│                                                                      │
│  All frames of Camera 0 ─┐                                           │
│                          ├──→ opt_id = 0 ──→ θ₀, ρ₀ (shared 6-DOF increments)│
│  All frames of Camera 0 ─┘                                           │
│                                                                      │
│  All frames of Camera 1 ─┐                                           │
│                          ├──→ opt_id = 1 ──→ θ₁, ρ₁ (shared 6-DOF increments)│
│  All frames of Camera 1 ─┘                                           │
│                                                                      │
│  Each iteration:                                                     │
│  1. Randomly sample one frame (cam_id=c, frame=t)                      │
│  2. Render I'_{c,t} using θ_c, ρ_c                                   │
│  3. Compute L_total, backward()                                      │
│  4. calib_optimizer[c].step() updates θ_c, ρ_c                      │
│  5. SE3_exp([ρ_c, θ_c]) converted to 4×4 matrix                       │
│  6. accum_P[c] = ΔT × accum_P[c] (incremental accumulation)          │
│  7. θ_c ← 0, ρ_c ← 0 (reset for next iteration)                      │
│                                                                      │
│  Key: Increment reset keeps se(3) increments near the origin,       │
│       small-angle approximation is more accurate, gradients more stable│
└──────────────────────────────────────────────────────────────────────┘
```

**Key implementation points** (`train.py` + `scene/poses.py`):

1. **Shared extrinsics**: When `use_rig=True`, `opt_cam_num = number of physical cameras` (e.g., 4 for KITTI-360). All frames from the same camera share one `cam_rot_deltas[opt_id]` and `cam_trans_deltas[opt_id]`
2. **Incremental accumulation**: Each iteration converts increments to 4×4 matrices via `SE3_exp`, accumulates to global pose `accum_P`, then resets increment parameters to zero
3. **Warmup mechanism**: `min_viewpoint_cycle=5`, ensuring all training viewpoints are seen by the scene optimizer at least 5 rounds before releasing pose parameters
4. **Per-camera independent optimizers**: `calib_optimizer` is a list; each camera/rig has its own Adam optimizer

### 3.5 Adaptive Voxel Control (AVC)

**Paper Section III-E**: Automatically determines voxel size ε* via binary search.

```
ε* = argmin_ε |V(ε) - V_target|
V_target = β × D_traj
D_traj = Σ_{t=1}^{T-1} ||T^L_{t+1} - T^L_t||_2
```

```
┌──────────────────────────────────────────────────────────────────────┐
│                      AVC Binary Search Algorithm                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input: LiDAR point cloud P, target voxel count V_target, β=5000     │
│                                                                      │
│  1. Compute trajectory length:                                       │
│     D_traj = Σ ||T^L_{t+1}[:3,3] - T^L_t[:3,3]||                    │
│                                                                      │
│  2. Compute target: V_target = β × D_traj                            │
│                                                                      │
│  3. Binary search ε ∈ [0.1, 0.5]:                                    │
│     while |V(ε) - V_target| / V_target > 5%:                         │
│       mid = (lo + hi) / 2                                             │
│       n = count_voxels(P, mid)                                        │
│       if n > V_target: lo = mid  (voxels too dense, increase ε)       │
│       else:            hi = mid  (voxels too sparse, decrease ε)      │
│                                                                      │
│  4. Output: optimal voxel size ε*                                    │
│                                                                      │
│  Examples:                                                           │
│  KITTI-360 (long straight line, D_traj≈200m): ε* ≈ 0.15-0.20        │
│  FAST-LIVO2 (small indoor scene, D_traj≈50m): ε* ≈ 0.10             │
└──────────────────────────────────────────────────────────────────────┘
```

Importance of AVC: Too small ε leads to overly dense anchors (unstable optimization); too large ε reduces geometric coverage (accuracy degradation).

### 3.6 Loss Functions

**Paper Equations (8)-(9)**:

```
L_total = (1 - λ_dssim) · L_photo + λ_dssim · L_ssim + λ_scale · L_scale
L_photo = L1(I', I)   or   L2(I', I)
L_ssim  = 1 - SSIM(I', I)
L_scale = (1/|V|) · Σ_{i∈V} max(max(s_i)/min(s_i) - σ, 0)
```

Default hyperparameters:
- `lambda_dssim = 0.2` (consistent with standard 3DGS)
- `lambda_scale = 1.0`
- `scale_regularizer (σ) = 10`
- `loss_type = "l1"`

`L_scale` penalizes overly anisotropic Gaussians (max/min ratio exceeding σ=10), preventing degenerate elongated Gaussians from destabilizing optimization. This is a new regularization term in TLC-Calib; standard 3DGS and Scaffold-GS do not have it.

### 3.7 Pose Learning Rate Scheduling

**TLC-Calib uses a cosine annealing scheduler** (unlike the exponential decay in standard 3DGS):

```
lr(t) = lr_final + 0.5 × (lr_init - lr_final) × (1 + cos(π × t/T))
```

| Parameter | Initial | Final | Schedule Type | Decay Ratio |
|------|--------|------|---------|--------|
| Rotation lr | 2×10⁻³ | 2×10⁻⁴ | Cosine annealing | 0.1× |
| Translation lr | 5×10⁻³ | 5×10⁻⁴ | Cosine annealing | 0.1× |
| Scene parameters | Various | Various | **Exponential decay** | - |

Scene weight decay strategy: `weight_decay=1e-2` for the first 15K iterations, then disabled (set to 0), ensuring no overfitting in the first half while allowing fine adjustment in the second half.

**Why cosine annealing for pose but exponential decay for scene?** Cosine annealing decays more slowly at the end of training, allowing pose parameters to continue fine-tuning. Exponential decay is more suitable for scene parameters, which need to stabilize faster.

### 3.8 From-LiDAR Initialization

Initialization protocol: Camera poses are derived from LiDAR poses using a base camera-to-lidar transformation matrix.

```
BASE_CAMERA_TO_LIDAR = [         Additional yaw offsets per dataset:
    [0,  0,  1,  0],            ┌────────────────────────────────┐
    [-1, 0,  0,  0],            │ KITTI-360: cam2=+90°, cam3=-90°│
    [0,  -1, 0,  0],            │ Waymo:     cam1=+45°, cam2=-45°│
    [0,  0,  0,  1]             │            cam3=+90°, cam4=-90°│
]                                │ FAST-LIVO2: no offset          │
                                 └────────────────────────────────┘
Camera z-axis → LiDAR x-axis (forward)
Camera x-axis → LiDAR -y-axis (left)
Camera y-axis → LiDAR -z-axis (down)
```

This initialization produces the challenging scenario mentioned in the paper where "translation error can reach 1.2m".

---

## 4. Deep Analysis of Pose Gradient Propagation in the CUDA Rasterizer

### 4.1 Custom Rasterizer Architecture

TLC-Calib uses `diff-gaussian-rasterization-w-pose`, an improved version of the standard 3DGS rasterizer. The core modification adds gradient computation for pose increments `τ = [ρ, θ]` (6-dimensional se(3) vector) in CUDA kernels.

```
File structure:
submodules/diff-gaussian-rasterization-w-pose/
├── cuda_rasterizer/
│   ├── forward.cu         # Forward: 3D→2D projection, alpha compositing
│   ├── backward.cu        # Backward: gradient computation (core modification)
│   ├── math.h             # SE3/SO3 Lie group operations (new)
│   ├── auxiliary.h        # Auxiliary functions
│   ├── forward.h / backward.h
│   └── rasterizer.h / rasterizer_impl.cu
├── rasterize_points.cu    # PyTorch ↔ CUDA interface
└── diff_gaussian_rasterization_w_pose/
    └── __init__.py        # Python binding, extracts grad_theta/grad_rho
```

### 4.2 Forward Pass

The forward pass is nearly identical to standard 3DGS. Key steps:

1. **Preprocessing** (`forward.cu::preprocessCUDA`): Project 3D Gaussians to the 2D image plane
   - Transform Gaussian centers to camera coordinates using `viewmatrix` (world-to-camera transform)
   - Compute 2D covariance matrix (EWA Splatting)
   - Invert to obtain conic matrix
   
2. **Rendering** (`forward.cu::renderCUDA`): Alpha compositing
   - Layer-by-layer blending of depth-sorted Gaussians
   - Standard 3DGS front-to-back sorted rendering

**Key difference**: Pose increments `theta/rho` do not directly participate in forward computation (when increments are zero the transform is identity); gradients are computed analytically via Jacobian in the backward pass.

### 4.3 Backward Pass — Four Gradient Paths

Pose gradient `dL/dτ` accumulates through **four** independent paths. Each Gaussian contributes a 6-dimensional gradient `dL_dtau[6*idx + 0..5]`:

```
┌──────────────────────────────────────────────────────────────────────────┐
│              Four Propagation Paths for Pose Gradient dL/dτ              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Path ①: renderCUDA (backward)                                           │
│  ┌─────────────────────┐                                                 │
│  │ dL/dPixel            │                                                │
│  │   ↓ alpha blending   │                                                │
│  │ dL/dColor, dL/dAlpha │                                                │
│  │   ↓                  │                                                │
│  │ dL/dMean2D           │──→ passed to Path ②                             │
│  │ dL/dConic2D          │──→ passed to Path ③                             │
│  │ dL/dDepth            │──→ passed to Path ④                             │
│  └─────────────────────┘                                                 │
│                                                                          │
│  Path ②: preprocessCUDA — 2D projected mean → pose                       │
│  ┌───────────────────────────────────────────────────┐                    │
│  │ dL/dτ += dL/dMean2D · ∂Mean2D/∂p_C · ∂p_C/∂τ     │                   │
│  │                                                    │                   │
│  │ p_C = T_CW · p_W  (3D point in camera coordinates)│                   │
│  │ ∂p_C/∂ρ = I₃ₓ₃  (translation acts directly)       │                   │
│  │ ∂p_C/∂θ = -[p_C]× (skew-symmetric matrix)          │                   │
│  │                                                    │                   │
│  │ Using analytic Jacobian of projection matrix:      │                   │
│  │ d_proj_dp_C = [α·a,  0,   β·e]                     │                   │
│  │               [0,    α·b, γ·e]                      │                   │
│  │ where α=1/w, β=-x/w², γ=-y/w²                       │                   │
│  └───────────────────────────────────────────────────┘                    │
│                                                                          │
│  Path ③: computeCov2DCUDA — 2D covariance → pose                         │
│  ┌───────────────────────────────────────────────────┐                    │
│  │ dL/dτ += dL/dΣ²ᴰ · ∂Σ²ᴰ/∂t_cam · ∂t_cam/∂τ      │                   │
│  │        + dL/dΣ²ᴰ · ∂Σ²ᴰ/∂W · ∂W/∂θ               │                   │
│  │                                                    │                   │
│  │ Translation part:                                   │                   │
│  │   dp_C_d_rho = I₃ₓ₃                                │                   │
│  │   dp_C_d_theta = -[p_C]×                            │                   │
│  │                                                    │                   │
│  │ Rotation part (per-column derivative of rotation): │                   │
│  │   ∂W/∂θ = [-[r₁]×, -[r₂]×, -[r₃]×]                │                   │
│  │   where r₁,r₂,r₃ are the three columns of R         │                   │
│  └───────────────────────────────────────────────────┘                    │
│                                                                          │
│  Path ④: preprocessCUDA — depth → pose                                    │
│  ┌───────────────────────────────────────────────────┐                    │
│  │ dL/dτ += dL/dDepth · ∂depth/∂p_C · ∂p_C/∂τ        │                   │
│  │                                                    │                   │
│  │ depth = p_C.z                                       │                   │
│  │ ∂depth/∂p_C = [0, 0, 1]                             │                   │
│  │ ∂p_C/∂ρ and ∂p_C/∂θ same as Path ②                  │                   │
│  └───────────────────────────────────────────────────┘                    │
│                                                                          │
│  Path ⑤ (retained but inactive): computeColorFromSH — SH color → pose    │
│  ┌───────────────────────────────────────────────────┐                    │
│  │ Active only when using spherical harmonics         │                   │
│  │ TLC-Calib uses MLP precomputed colors, SH=None     │                   │
│  │ Code retains this path for generality                │                   │
│  │                                                    │                   │
│  │ dL/dτ[0:3] += -dL/dmean (translation direction)    │                   │
│  └───────────────────────────────────────────────────┘                    │
│                                                                          │
│  Final: dL_dtau[6*idx + 0..5] = Path② + Path③ + Path④ (+ Path⑤ if SH)    │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Gradient Aggregation and Python Binding

Per-point gradients from all Gaussians are aggregated into global pose gradients at the Python layer (`__init__.py`):

```python
# Backward output grad_tau: [P, 6] → sum → [6]
grad_tau = torch.sum(grad_tau.view(-1, 6), dim=0)
grad_rho = grad_tau[:3].view(1, -1)     # translation gradient [1, 3]
grad_theta = grad_tau[3:].view(1, -1)   # rotation gradient [1, 3]
```

This implements the `Σ_i` in Paper Equation (2), summing gradient contributions from all visible Gaussians.

### 4.5 Lie Algebra Mathematical Foundations (CUDA Implementation)

`math.h` implements complete SE(3)/SO(3) Lie group/Lie algebra operations:

```
┌────────────────────────────────────────────────────────────────────┐
│                    Lie Group/Lie Algebra Operations in math.h     │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  struct SO3:                                                       │
│    ├── Exp(θ): Rodrigues formula                                   │
│    │   R = I + sin(||θ||)/||θ|| · [ω]×                             │
│    │     + (1-cos(||θ||))/||θ||² · [ω]×²                          │
│    ├── data(): returns mat33 rotation matrix                       │
│    └── multiplication operator: SO3 × SO3                          │
│                                                                    │
│  struct SE3:                                                       │
│    ├── Constructed from viewmatrix float[16]                       │
│    ├── R(): returns SO3 rotation                                   │
│    ├── t(): returns float3 translation                             │
│    ├── Exp(τ): T = [R, V·ρ; 0, 1]                                  │
│    │   where V is the SO3 left Jacobian                            │
│    └── operator*: SE3 applied to float3 point                        │
│                                                                    │
│  struct mat33:                                                     │
│    ├── identity(): identity matrix                                 │
│    ├── skew_symmetric(v): skew-symmetric matrix [v]×               │
│    ├── transpose(): transpose                                      │
│    └── matrix-vector multiplication                                │
│                                                                    │
│  Corresponding Python implementation (utils/pose_utils.py):        │
│    _skew_symmetric(v) → mat33::skew_symmetric                      │
│    _SO3_exp(θ)        → SO3::Exp                                    │
│    _SO3_left_jacobian(θ) → used internally in SE3::Exp             │
│    SE3_exp(τ)         → SE3::Exp                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 4.6 Complete Gradient Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Complete System Gradient Flow                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  L_total = (1-λ)·L1 + λ·(1-SSIM) + λ_s·L_scale                               │
│       │                                                                      │
│       ▼ .backward()                                                          │
│  ┌─────────────────────────────────────────────────────┐                      │
│  │ PyTorch Autograd                                     │                     │
│  │  ├─→ dL/dImage  ──→ CUDA renderCUDA backward         │                     │
│  │  │                    ├─→ dL/dMean2D (per-gaussian)   │                     │
│  │  │                    ├─→ dL/dConic2D                 │                     │
│  │  │                    ├─→ dL/dColor                   │                     │
│  │  │                    ├─→ dL/dOpacity                 │                     │
│  │  │                    └─→ dL/dDepth                   │                     │
│  │  │                                                    │                     │
│  │  ├─→ CUDA computeCov2DCUDA                            │                     │
│  │  │     ├─→ dL/dMean3D (from cov path)                 │                     │
│  │  │     ├─→ dL/dCov3D                                  │                     │
│  │  │     └─→ dL/dτ (Path③: covariance→pose)             │                     │
│  │  │                                                    │                     │
│  │  └─→ CUDA preprocessCUDA                              │                     │
│  │        ├─→ dL/dMean3D (from proj path)                │                     │
│  │        ├─→ dL/dScale, dL/dRot                         │                     │
│  │        ├─→ dL/dτ (Path②: 2D mean→pose)                  │                     │
│  │        └─→ dL/dτ (Path④: depth→pose)                  │                     │
│  └─────────────────────────┬───────────────────────────┘                      │
│                            │                                                  │
│            ┌───────────────┴───────────────┐                                  │
│            ▼                               ▼                                  │
│  ┌─────────────────────┐     ┌─────────────────────────┐                      │
│  │ Scene Optimizer      │     │ Calib Optimizer          │                     │
│  │ (AdamW, per-param)   │     │ (Adam, per-camera)       │                     │
│  │                      │     │                          │                     │
│  │ MLP params ← grad    │     │ θ_c ← grad_theta        │                     │
│  │ anchor_feat ← grad   │     │ ρ_c ← grad_rho          │                     │
│  │ offset ← grad        │     │                          │                     │
│  │ scaling ← grad       │     │ ↓ SE3_exp([ρ_c, θ_c])   │                     │
│  │                      │     │ accum_P[c] = ΔT × P[c]  │                     │
│  │ (anchor lr=0 frozen) │     │ θ_c ← 0, ρ_c ← 0       │                     │
│  └─────────────────────┘     └─────────────────────────┘                      │
│                                                                              │
│  dL/dScale → L_scale regularization also contributes gradients via this path │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. How the Refine Stage Works

### 5.1 Design Motivation

After calibration completes (30K iterations), the scene representation is well aligned with optimized extrinsics. The Refine stage aims to further improve scene rendering quality (NVS performance) under **frozen pose** conditions.

### 5.2 Stage Comparison

```
┌──────────────────────────────────────────────────────────────────────┐
│                  Main Training vs Refine Stage Comparison            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────── Main Training Stage (0-30K) ────────────┐               │
│  │ • opt_pose = True (pose optimizable)              │               │
│  │ • Densification: 500-15K (prune only)            │               │
│  │ • Weight decay: 0-15K: 1e-2, after 15K: 0        │               │
│  │ • Pose lr: cosine annealing (high→low)           │               │
│  │ • Save: iteration 30000                         │               │
│  └────────────────────────────────────────────┘                       │
│                                                                      │
│  ┌─────────── Refine Stage (30K-40K) ──────────┐                       │
│  │ • opt_pose = False (pose frozen!)           │                       │
│  │ • Densification: 30500-39000                │                       │
│  │   └─ Re-statistics gradients, allow anchor growing│               │
│  │ • Weight decay: remains 0                   │                       │
│  │ • Pose lr: N/A (frozen)                     │                       │
│  │ • Save: iteration 40000                     │                       │
│  └────────────────────────────────────────────┘                       │
│                                                                      │
│  Key code:                                                           │
│  if refine_phase:                                                    │
│      opt.start_stat = opt.refine_start        # 30000                │
│      opt.update_from = opt.refine_start + 500 # 30500                │
│      opt.update_until = opt.refine_end - 1000 # 39000                │
│      opt.opt_pose = False                     # freeze pose          │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.3 Refine Stage Densification

In `adjust_anchor`, when `refine=True`, `anchor_growing` is enabled:

1. Compute normalized offset gradient accumulation
2. Add new anchors in regions with large gradients (insufficient geometric detail)
3. Multi-level growth: `update_depth=3`, threshold increases per level `cur_threshold = threshold × (hierachy_factor/2)^i`
4. Random sampling to avoid overcrowding: `rand_mask > 0.5^(i+1)`
5. Deduplication: new anchors do not overlap existing ones (grid coordinate check)

---

## 6. Code Differences vs. Scaffold-GS / MonoGS

### 6.1 Three-Project Architecture Comparison Table

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   Scaffold-GS / MonoGS / TLC-Calib Comparison            │
├────────────────┬──────────────┬──────────────┬───────────────────────────┤
│     Aspect     │ Scaffold-GS  │   MonoGS     │      TLC-Calib            │
├────────────────┼──────────────┼──────────────┼───────────────────────────┤
│ Scene repr.    │ Anchor+Aux   │ Standard 3DGS│ Anchor+Aux (Scaffold)     │
│ Anchor position│ Learnable    │ N/A          │ Frozen (lr=0)             │
│ Pose optimization│ Not supported│ Supported(SLAM)│ Supported(calibration)  │
│ Pose type      │ N/A          │ Per-frame cam│ Rig extrinsics(cam-to-lidar)│
│ Reference frame│ COLMAP       │ First camera │ LiDAR coordinate frame    │
│ Rasterizer     │ Standard     │ Custom(w-pose)│ Custom(w-pose, MonoGS)    │
│ Color repr.    │ MLP          │ SH           │ MLP (Scaffold)            │
│ Gradient paths │ No pose grad │ mean+cov+SH  │ mean+cov+depth(no SH)     │
│ Optimizer      │ Adam         │ Adam         │ AdamW(scene)+Adam(pose) │
│ Loss functions │ L1+DSSIM     │ L1+DSSIM     │ L1+DSSIM+L_scale          │
│ Voxel control  │ Manual       │ N/A          │ AVC adaptive              │
│ Init source    │ COLMAP cloud │ Depth map    │ LiDAR point cloud         │
│ Weight decay   │ Not used     │ Not used     │ AdamW phased decay        │
│ Multi-camera   │ Not supported│ Not supported│ Multi-camera Rig          │
│ Convergence mon.│ None         │ None         │ convergence_score         │
│ Training stages│ Single stage │ Single stage │ Dual (Main+Refine)        │
│ Visualization  │ None         │ None         │ Viser Web Viewer          │
│ Application    │ NVS          │ SLAM         │ LiDAR-Camera extrinsic calib│
└────────────────┴──────────────┴──────────────┴───────────────────────────┘
```

### 6.2 TLC-Calib Unique Innovations

1. **Anchor freezing strategy**: Ensures global scale from LiDAR does not drift during optimization
2. **Rig optimization**: All frames from the same physical camera share one extrinsic, stronger constraint
3. **AVC**: Automatically determines anchor density based on LiDAR trajectory length
4. **Scale regularization L_scale**: Prevents anisotropic degenerate Gaussians from hindering pose convergence
5. **Warmup mechanism**: Stabilize scene representation first, then release pose optimization
6. **AdamW weight decay**: Scene optimizer uses AdamW to prevent overfitting
7. **Depth gradient path**: New pose gradient propagation for depth rendering in backward.cu
8. **Convergence monitoring**: Real-time `||τ||` norm computation for convergence assessment

---

## 7. Data Preprocessing and Custom Dataset Integration

### 7.1 Data Directory Structure

```
<dataset_scene>/
├── images/           # Image sequences per camera
│   ├── image_00/     # Camera 0: 000000.png, 000001.png, ...
│   ├── image_01/     # Camera 1
│   └── ...
├── lidar/            # LiDAR data
│   ├── map.ply       # Scene-level aggregated point cloud (required)
│   └── input_0.15.ply # AVC downsampled point cloud (auto-generated)
├── params/           # Calibration parameters
│   ├── intrinsics.txt        # Per-camera intrinsics (3×3 matrix)
│   ├── lidars.txt             # LiDAR pose sequence (N×16 floats)
│   ├── cams_to_lidar_gt.txt   # GT camera-to-lidar extrinsics
│   └── timestamps.txt        # Timestamps (optional, for time offset)
└── cams_to_lidar_gt.txt       # Backup to output directory
```

### 7.2 Parameter File Format Details

```
intrinsics.txt:           lidars.txt:                cams_to_lidar_gt.txt:
# CAM 0:                  r00 r01 r02 tx ...         0 r00 r01 ... tz 0 0 0 1
fx  0   cx                r10 r11 r12 ty ...         1 r00 r01 ... tz 0 0 0 1
0   fy  cy                ...  (16 floats per line,   (cam_id + 16 floats)
0   0   1                  one 4×4 l2w matrix)
# CAM 1:
...
```

### 7.3 Complete Data Preprocessing Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│              read_custom_scene_info() Processing Flow                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ① Read all parameter files                                          │
│     ├── cams_to_lidar_gt.txt → cams2lidar (GT extrinsics)            │
│     ├── lidars.txt → lidar_extrinsics [N, 4, 4]                      │
│     ├── intrinsics.txt → cam_intrinsics {cam_id: 3×3}                │
│     └── timestamps.txt → lidar_timestamps [N] (optional)             │
│                                                                      │
│  ② Compute trajectory length                                         │
│     D_traj = Σ ||T^L_{t+1}[:3,3] - T^L_t[:3,3]||₂                  │
│                                                                      │
│  ③ Copy LiDAR poses to each camera                                   │
│     lidar_extrinsics: [N,4,4] → [M,N,4,4]  (M=number of cameras)     │
│                                                                      │
│  ④ Time offset handling (if time_offset > 0)                         │
│     Random ± direction time offset per camera                        │
│     Adjust LiDAR poses via Slerp rotation + linear translation interp.│
│                                                                      │
│  ⑤ From-LiDAR initialization (if from_lidar=True)                    │
│     init_c2l = BASE_CAMERA_TO_LIDAR × yaw_transform(cam_id)         │
│     w2c = inv(lidar_pose × init_c2l)                                 │
│                                                                      │
│  ⑥ Noise injection (if t_noise > 0 or r_noise > 0)                   │
│     Axis-angle rotation noise + random-direction translation noise     │
│                                                                      │
│  ⑦ Build CameraInfo list                                              │
│     Per frame: cam_id, uid, GT_R/t, init_R/t, lidar_R/t, intrinsics  │
│                                                                      │
│  ⑧ Train/test split                                                  │
│     llffhold=2: every 2nd frame as test                              │
│                                                                      │
│  ⑨ Adaptive voxelization (if adaptive_voxel=True)                    │
│     voxel_size = compute_voxel_size(pc, β × D_traj)                  │
│                                                                      │
│  ⑩ Open3D voxel downsampling                                         │
│     voxel_down_sample_and_trace → retain representative points       │
│     Store as lidar/input_{voxel_size}.ply                            │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.4 Custom Dataset Integration Guide

To integrate a new dataset:

1. **Prepare calibration files** in the `params/` directory
2. **Aggregate LiDAR point cloud** as `lidar/map.ply` (using SLAM or ICP alignment)
3. **Synchronize image and LiDAR frames** (ensure frame indices match; images named `000000.png`, `000001.png`, ...)
4. Add yaw configuration for the new dataset in `DATASET_CAMERA_YAW_DEGREES` in `utils/pose_utils.py`
5. Specify `--dataset <your_dataset_name>` at runtime

---

## 8. NVS Evaluation Module Workflow

### 8.1 Evaluation Strategy

Core philosophy of NVS evaluation: **Verify calibration quality using the standard 3DGS pipeline**.

All methods (including baselines) retrain the scene using the same 3DGS pipeline. The only difference is the input camera-to-lidar extrinsics. This ensures NVS performance differences come entirely from calibration accuracy.

### 8.2 Evaluation Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Complete NVS Evaluation Pipeline                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  metrics_nvs.py::evaluate()                                          │
│  │                                                                   │
│  ├─① Read calibration output                                        │
│  │   model_path/config.yml → source_path                             │
│  │   model_path/point_cloud/iteration_30000/                         │
│  │     ├── cams_to_lidar.txt (calibrated extrinsics)                 │
│  │     └── time_offset.txt (time offset, optional)                   │
│  │                                                                   │
│  ├─② Call nvs_eval/train.py                                          │
│  │   python nvs_eval/train.py                                        │
│  │     -s <source_path>                                              │
│  │     -m <model_path>/nvs_eval/ours_30000                           │
│  │     --prior_pose_path <cams_to_lidar.txt>                         │
│  │     --prior_time_path <time_offset.txt>                           │
│  │     --eval --voxel 0.1 --llffhold 2                               │
│  │     --iterations 30000                                            │
│  │                                                                   │
│  ├─③ nvs_eval/train.py internals                                      │
│  │   ├── Standard 3DGS (no pose optimization)                        │
│  │   ├── Standard densification (split/clone/prune)                    │
│  │   ├── SH color representation (not MLP)                           │
│  │   ├── Train 30K iterations                                        │
│  │   ├── Render test set: renders/ + gt/                             │
│  │   └── Compute PSNR, SSIM, LPIPS                                   │
│  │                                                                   │
│  └─④ Save results                                                     │
│      model_path/nvs_results.json:                                    │
│      {                                                               │
│        "ours_30000": {"PSNR": 26.39, "SSIM": 0.85, "LPIPS": 0.09},  │
│        "source_path": "...",                                         │
│        "prior_pose_path": "..."                                      │
│      }                                                               │
└──────────────────────────────────────────────────────────────────────┘
```

### 8.3 Pose Evaluation

```
metrics_pose.py::evaluate():
  ① Read cams_to_lidar.txt (predicted rig)
  ② Read cams_to_lidar_gt.txt (GT rig)
  ③ Rotation error: arccos((trace(R_pred · R_gt^T) - 1) / 2) → degrees
  ④ Translation error: ||t_pred - t_gt||₂ → meters
  ⑤ Output rig_results.json
```

---

## 9. Web Viewer Real-Time Visualization Module

### 9.1 Architecture Design

TLC-Calib integrates a **viser + nerfview** based Web real-time visualizer, allowing browser-based monitoring of calibration progress during training.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Web Viewer Architecture                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    HTTP/WebSocket     ┌──────────────────────┐     │
│  │ Browser Client│ ◄──────────────────► │  viser.ViserServer    │     │
│  │ (WebGL render)│   port:8080          │  (Python backend)    │     │
│  └──────────────┘                        └────────┬─────────────┘     │
│                                                   │                  │
│                                          ┌────────▼─────────────┐     │
│                                          │  ViewerRenderer       │     │
│                                          │  (GPU render callback)│     │
│                                          │                      │     │
│                                          │  ① Receive camera state│     │
│                                          │  ② Construct virtual Camera│  │
│                                          │  ③ Call render()      │     │
│                                          │  ④ Return numpy image │     │
│                                          │  ⑤ Update Anchor/Offset│    │
│                                          │     point cloud display│     │
│                                          │  ⑥ Update camera frustums│   │
│                                          └──────────────────────┘     │
│                                                                      │
│  GUI Controls:                                                        │
│  ┌────────────────────────────────────────────────────────┐           │
│  │ Camera Views:    [Orthogonal View]                      │           │
│  │ Visibility:      [✓] Initial Points  [✓] Anchors       │           │
│  │                  [✓] Auxiliary        [✓] Opt Cams      │           │
│  │                  [✓] Ref Cams         Content: [Image▼] │           │
│  │ Scale:           Point Size [0.01━━━━0.1]               │           │
│  │                  Gaussian Scale [0.1━━━━2.0]            │           │
│  └────────────────────────────────────────────────────────┘           │
│                                                                      │
│  Visualization content:                                              │
│  • Initial LiDAR point cloud (blue, 50,150,255)                      │
│  • Anchor point cloud (green, 50,200,50) — real-time update          │
│  • Auxiliary offset points (red, 250,50,50) — real-time update       │
│  • Optimizing camera frustums (gray, clickable to jump)                │
│  • GT camera frustums (yellow, clickable to jump)                      │
│  • Each camera can display corresponding image                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.2 Integration with Training Loop

```python
# Viewer integration in train.py
viewer.lock.acquire()         # Acquire lock to prevent scene modification during render
tic = time.time()
# ... training iteration ...
viewer.lock.release()         # Release lock

# Update viewer state
viewer.state.num_train_rays_per_sec = rays/sec
viewer.update(iteration, num_train_rays_per_step)

# Keep viewer alive after training ends
if viewer is not None:
    time.sleep(1000000)  # Keep process running
```

Usage: `python train.py ... --viewer --port 8080`, then open `http://localhost:8080` in a browser.

---

## 10. nvs_eval/ Submodule: Standard 3DGS Evaluation Pipeline

### 10.1 Design Philosophy

`nvs_eval/` is an independent **standard 3DGS** implementation, completely different from TLC-Calib's Neural Gaussian architecture. This is the core of the paper's fair comparison strategy:

```
┌──────────────────────────────────────────────────────────────────────┐
│              nvs_eval/ vs TLC-Calib Main Code Comparison             │
├─────────────────┬──────────────────────┬─────────────────────────────┤
│     Aspect      │    nvs_eval/          │    TLC-Calib Main Code      │
├─────────────────┼──────────────────────┼─────────────────────────────┤
│ Gaussian type   │ Standard 3DGS (flat)  │ Anchor + Auxiliary          │
│ Color repr.     │ Spherical harmonics   │ MLP                         │
│ Pose optimization│ None                  │ SE(3) rig optimization      │
│ Rasterizer      │ Standard diff-gaussian-│ Custom -w-pose              │
│                 │ rasterization         │                             │
│ Densification   │ Standard (split/clone/ │ Anchor growing/pruning      │
│                 │ prune/opacity reset)  │                             │
│ Point cloud init│ Uses calibrated extrin.│ From LiDAR point cloud      │
│ Purpose         │ Evaluate calibration   │ Joint calibration+scene repr│
│ Input extrinsics│ prior_pose_path        │ From-LiDAR initialization   │
│ iterations      │ 30000                 │ 30000 + 10000(refine)       │
│ Train/test split│ llffhold=2            │ llffhold=2                  │
└─────────────────┴──────────────────────┴─────────────────────────────┘
```

### 10.2 nvs_eval/train.py Core Pipeline

```
nvs_eval/train.py:
  ① GaussianModel(sh_degree=0)        # Standard 3DGS
  ② Scene(dataset, gaussians)          # Load data (using calibrated extrinsics)
  ③ training_setup(opt, fix_xyz=False) # xyz learnable (unlike TLC-Calib freeze)
  ④ for iteration in range(30000):
       viewpoint_cam = random_camera()
       render_pkg = render(cam, gaussians, pipe, bg)
       Ll1 = l1_loss(image, gt)
       loss = (1-λ) * Ll1 + λ * (1-ssim)  # Standard loss (no L_scale)
       loss.backward()
       if iteration < densify_until:
           densify_and_prune()          # Standard 3DGS densification
       optimizer.step()
  ⑤ render_sets(test_cameras)          # Render test set
  ⑥ evaluate(PSNR, SSIM, LPIPS)       # Compute metrics
```

**Key point**: nvs_eval performs no pose optimization, relying entirely on input extrinsics. Therefore NVS quality directly reflects calibration accuracy.

---

## 11. full_eval.py Batch Evaluation System

### 11.1 Architecture Design

```
┌──────────────────────────────────────────────────────────────────────┐
│                   full_eval.py Batch Evaluation Pipeline             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Command: python full_eval.py -d /path/to/datasets -o ./outputs      │
│          --datasets kitti-360 waymo --repeat 3                       │
│                                                                      │
│  ┌──────────────────────────────────────────┐                        │
│  │ discover_scenes()                         │                        │
│  │  /datasets/                               │                        │
│  │    ├── kitti-360/                         │                        │
│  │    │   ├── scene_0000/ → discovered       │                        │
│  │    │   ├── scene_0001/ → discovered       │                        │
│  │    │   └── scene_0002/ → discovered       │                        │
│  │    └── waymo/                             │                        │
│  │        ├── segment_001/ → discovered      │                        │
│  │        └── segment_002/ → discovered      │                        │
│  └──────────────────────┬───────────────────┘                        │
│                         │                                             │
│  For each scene × each repeat:                                        │
│  ┌──────────────────────▼───────────────────┐                        │
│  │ run_full_eval():                          │                        │
│  │  ① python train.py -s <source> -m <model> │                        │
│  │     --eval --from_lidar --use_rig         │                        │
│  │     --opt_pose --pose_scheduler           │                        │
│  │     --adaptive_voxel                      │                        │
│  │                                           │                        │
│  │  ② python metrics_pose.py -m <model>      │                        │
│  │     → rig_results.json                    │                        │
│  │                                           │                        │
│  │  ③ python metrics_nvs.py -m <model>       │                        │
│  │     → nvs_results.json                    │                        │
│  │                                           │                        │
│  └──────────────────────┬───────────────────┘                        │
│                         │                                             │
│  ┌──────────────────────▼───────────────────┐                        │
│  │ aggregate_results():                      │                        │
│  │  ├── aggregate_pose_results()             │                        │
│  │  │   per-camera average Rot/Trans Err     │                        │
│  │  │   → exp_rig_results.json               │                        │
│  │  │                                        │                        │
│  │  ├── aggregate_nvs_results()              │                        │
│  │  │   average PSNR/SSIM/LPIPS             │                        │
│  │  │   → exp_nvs_results.json               │                        │
│  │  │                                        │                        │
│  │  └── aggregate_train_results()            │                        │
│  │      average Peak/Avg Memory, Train Time  │                        │
│  │      → train_results.json                 │                        │
│  └──────────────────────┬───────────────────┘                        │
│                         │                                             │
│  ┌──────────────────────▼───────────────────┐                        │
│  │ write_all_dataset_summaries():            │                        │
│  │  outputs/                                 │                        │
│  │    ├── kitti-360/                         │                        │
│  │    │   ├── scene_0000/                    │                        │
│  │    │   │   ├── eval_01/                   │                        │
│  │    │   │   ├── eval_02/                   │                        │
│  │    │   │   ├── eval_03/                   │                        │
│  │    │   │   ├── exp_rig_results.json       │                        │
│  │    │   │   ├── exp_nvs_results.json       │                        │
│  │    │   │   └── train_results.json         │                        │
│  │    │   └── full_eval_results.json  ← summary│                      │
│  │    └── waymo/                             │                        │
│  │        └── full_eval_results.json  ← summary│                      │
│  └──────────────────────────────────────────┘                        │
└──────────────────────────────────────────────────────────────────────┘
```

### 11.2 Aggregation Logic

`full_eval.py` supports three levels of aggregation:

1. **Scene-level aggregation**: Average across multiple repeat runs (`--repeat N`)
2. **Dataset-level summary**: Collect results across all scenes in a dataset
3. **Aggregation-only mode**: `--aggregate_path` skips training, re-aggregates existing results

**Default training parameters** (`DEFAULT_TRAIN_ARGS`):

```python
["--eval", "--from_lidar", "--use_rig", "--opt_pose", "--pose_scheduler", "--adaptive_voxel"]
```

These are the standard configurations used in paper experiments.

---

## 12. Consistency Between Paper Experiments and Code Hyperparameters

### 12.1 Hyperparameter Comparison Table

| Paper Description | Paper Value | Code Default | File Location | Consistency |
|---------|--------|-----------|---------|--------|
| Total iterations | 30K | `iterations=30000` | arguments/__init__.py:113 | **Consistent** |
| Refine iterations | 10K | `refine_iterations=10000` | arguments/__init__.py:119 | **Consistent** |
| D-SSIM weight λ | 0.2 | `lambda_dssim=0.2` | arguments/__init__.py:182 | **Consistent** |
| Scale reg. weight | 1.0 | `lambda_scale=1.0` | arguments/__init__.py:183 | **Consistent** |
| Scale reg. threshold σ | 10 | `scale_regularizer=10.0` | arguments/__init__.py:184 | **Consistent** |
| AVC β | 5000 | `avc_beta=5000` | arguments/__init__.py:76 | **Consistent** |
| Rotation lr init | 2×10⁻³ | `calib_rot_lr_init=0.002` | arguments/__init__.py:126 | **Consistent** |
| Rotation lr final | 2×10⁻⁴ | `calib_rot_lr_final=0.0002` | arguments/__init__.py:127 | **Consistent** |
| Translation lr init | 5×10⁻³ | `calib_trans_lr_init=0.005` | arguments/__init__.py:128 | **Consistent** |
| Translation lr final | 5×10⁻⁴ | `calib_trans_lr_final=0.0005` | arguments/__init__.py:129 | **Consistent** |
| Warmup cycles | 5 | `min_viewpoint_cycle=5` | arguments/__init__.py:114 | **Consistent** |
| Feature dimension | 32 | `feat_dim=32` | arguments/__init__.py:58 | **Consistent** |
| Auxiliary Gaussian count K | 10 | `n_offsets=10` | arguments/__init__.py:59 | **Consistent** |
| Voxel size | Adaptive | `voxel_size=0.1` (default/fallback) | arguments/__init__.py:57 | **Consistent** |
| Test split | llffhold=2 | `llffhold=2` | arguments/__init__.py:56 | **Consistent** |
| SH degree | 0 | `sh_degree=0` | arguments/__init__.py:55 | **Consistent** |
| Anchor lr | 0 (frozen) | `position_lr_init=0.0` | arguments/__init__.py:136 | **Consistent** |
| Weight decay | 1e-2→0 | `scene_weight_decay=1e-2` | arguments/__init__.py:132 | **Consistent** |
| Decay cutoff | 15K | `scene_decay_until=15000` | arguments/__init__.py:133 | **Consistent** |
| Densification range | 500-15K | `start_stat=500, update_until=15000` | arguments/__init__.py:187-190 | **Consistent** |
| NVS eval voxel | 0.1 | `DEFAULT_VOXEL_SIZE=0.1` | metrics_nvs.py:23 | **Consistent** |

**Conclusion**: Code default hyperparameters are fully consistent with paper experimental settings; paper results can be reproduced directly.

### 12.2 Hyperparameters Present in Code but Not Explicitly Mentioned in Paper

| Parameter | Default | Description |
|------|--------|------|
| `lr_delay_mult` | 0.01 | Initial delay decay multiplier for pose lr |
| `update_depth` | 3 | Number of hierarchy levels for anchor growing |
| `update_hierachy_factor` | 4 | Threshold increase factor per hierarchy level |
| `update_init_factor` | 16 | Initial growing voxel multiplier |
| `success_threshold` | 0.8 | Access success rate threshold for pruning |
| `min_opacity` | 0.005 | Anchors below this value are pruned |
| `densify_grad_threshold` | 0.0002 | Gradient threshold for growing |
| `loss_type` | "l1" | Photometric loss type (supports l1/l2) |
| `appearance_dim` | 0 | Appearance embedding dimension (disabled by default) |
| `znear/zfar` | 0.01/100 | Near/far clipping planes |

---

## 13. Supplementary Analysis of Core Contributions

### 13.1 Loss Landscape Smoothing Mechanism

**Core insight from Paper Figure 1**:

```
┌──────────────────────────────────────────────────────────────────────┐
│                  Loss Landscape Smoothing Mechanism Comparison       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Standard 3DGS (3DGS-Calib):                                         │
│  ┌──────────────────────────────────┐                                │
│  │ Gaussians fixed on LiDAR points  │                                │
│  │ ↓                                │                                │
│  │ Each Gaussian adapts per view    │                                │
│  │ ↓                                │                                │
│  │ Scene overfits current pose      │                                │
│  │ ↓                                │                                │
│  │ Rugged loss landscape: multiple local minima│                      │
│  │     ╱╲  ╱╲╱╲  ╱╲                  │                                │
│  │   ╱    ╲╱    ╲╱    ╲              │                                │
│  │  ╱                    ╲           │                                │
│  │ (pose easily trapped in local minima)│                             │
│  └──────────────────────────────────┘                                │
│                                                                      │
│  TLC-Calib (Neural Gaussians):                                       │
│  ┌──────────────────────────────────┐                                │
│  │ MLP generates view-dependent aux Gaussians│                        │
│  │ ↓                                │                                │
│  │ Same anchor repr. smooth across views│                             │
│  │ ↓                                │                                │
│  │ Auxiliary Gaussians fill LiDAR blind zones│                        │
│  │ ↓                                │                                │
│  │ Smooth loss landscape: global optimum easier to reach│             │
│  │         ╱    ╲                     │                                │
│  │       ╱        ╲                   │                                │
│  │     ╱            ╲                 │                                │
│  │   ╱                ╲               │                                │
│  │ (pose stably converges to global optimum)│                          │
│  └──────────────────────────────────┘                                │
│                                                                      │
│  Analysis:                                                           │
│  ① MLP shared parameters → implicit correlation across views → generalization not overfitting│
│  ② Auxiliary Gaussians extend to LiDAR blind zones → more gradient signal → stronger constraints│
│  ③ Anchor positions frozen → global structure does not drift → more consistent pose update direction│
│  ④ Rig optimization → multi-frame shared extrinsics → per-frame noise averaged out│
└──────────────────────────────────────────────────────────────────────┘
```

### 13.2 Time Offset Modeling

Time offset modeling in Paper Section IV-B and ablation experiments:

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Time Offset Processing Pipeline                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Problem: LiDAR and camera timestamps may be unsynchronized        │
│  Example: time_offset = 50ms means camera is 50ms delayed vs LiDAR │
│                                                                      │
│  Processing (pose_utils.py):                                         │
│                                                                      │
│  ① With timestamps (timestamps.txt):                                 │
│     apply_time_offset_with_timestamps()                               │
│     Per frame target_time = timestamp + offset                       │
│     ├─ Target time between frames: Slerp rotation + linear translation interp.│
│     └─ Target time out of range: constant angular + linear velocity extrapolation│
│                                                                      │
│  ② Without timestamps:                                              │
│     apply_time_offset_uniform()                                       │
│     Assume constant frame rate (default 10Hz)                         │
│     timestamps = [0, 0.1, 0.2, ...] (seconds)                         │
│                                                                      │
│  ③ Configuration:                                                     │
│     --time_offset 50  # random ±50ms per camera                       │
│     Each camera independent random direction: offset_i = 50 × random_choice([-1, 1])│
│                                                                      │
│  Physical meaning:                                                     │
│  Image captured by camera at t+Δt corresponds to LiDAR pose at t   │
│  LiDAR pose must be adjusted to t+Δt for correct camera pose computation│
└──────────────────────────────────────────────────────────────────────┘
```

### 13.3 Warmup and Convergence Monitoring

```
┌──────────────────────────────────────────────────────────────────────┐
│                  Warmup and Convergence Monitoring Mechanism          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Warmup mechanism (train.py):                                         │
│  ┌───────────────────────────────────────────────┐                    │
│  │ warmup = True                                  │                    │
│  │ viewpoint_cycle = 0                            │                    │
│  │                                                │                    │
│  │ Whenever viewpoint_stack is exhausted:         │                    │
│  │   viewpoint_cycle += 1                          │                    │
│  │                                                │                    │
│  │ if opt.opt_pose:                                │                    │
│  │   if viewpoint_cycle > min_viewpoint_cycle(5):  │                    │
│  │     ★ Release pose optimization                 │                    │
│  │     calib_optimizer[cam].step()                  │                    │
│  │   else:                                         │                    │
│  │     ✗ Zero gradients, reset optimizer state     │                    │
│  │     calib_optimizer[cam].state = defaultdict()   │                    │
│  └───────────────────────────────────────────────┘                    │
│                                                                      │
│  Design rationale:                                                     │
│  • First 5 cycles: scene repr. (MLP, offset) learns from random init│
│  • Pose gradients unreliable while scene representation is unstable  │
│  • After 5 cycles: scene has basic repr. per view, pose gradients more reliable│
│  • Reset optimizer state: prevent Adam momentum from warmup affecting formal optimization│
│                                                                      │
│  Convergence monitoring (poses.py::update_delta_pose):               │
│  ┌───────────────────────────────────────────────┐                    │
│  │ tau = [ρ_c, θ_c]  # current increment          │                    │
│  │ convergence_score = ||tau||₂                    │                    │
│  │                                                │                    │
│  │ Intuition: smaller increment → smaller pose change → closer to convergence│
│  │ Usage: progress bar display, early stopping interface retained│    │
│  └───────────────────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────────────┘
```

### 13.4 AdamW Weight Decay Strategy

```
┌──────────────────────────────────────────────────────────────────────┐
│                    AdamW Weight Decay Strategy                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TLC-Calib uses AdamW (not Adam) as scene optimizer:                 │
│  self.optimizer = torch.optim.AdamW(l, lr=0.0, eps=1e-15,            │
│                                     weight_decay=1e-2)               │
│                                                                      │
│  Phased weight decay:                                                │
│  ┌─────────────────────────────────────────────────┐                  │
│  │ iteration 0 ──────── 15000 ──────── 30000       │                  │
│  │                                                  │                  │
│  │  weight_decay = 1e-2   │   weight_decay = 0     │                  │
│  │  (regularization prevents overfitting)│   (allows fine tuning)│   │
│  │                         │                        │                  │
│  │  ← update_scene_decay(0.0) at iter 15K →        │                  │
│  └─────────────────────────────────────────────────┘                  │
│                                                                      │
│  Why not always use weight decay?                                      │
│  • First half: MLP parameters need regularization against noisy init pose overfitting│
│  • Second half: pose largely converged, scene needs fine adaptation to final pose│
│  • Excessive decay suppresses scene representation, reducing rendering quality│
│                                                                      │
│  Why AdamW instead of Adam + L2?                                       │
│  AdamW decouples weight decay from adaptive learning rate,            │
│  better regularization effect on MLP and similar parameters           │
└──────────────────────────────────────────────────────────────────────┘
```

### 13.5 Noise Injection and Robustness Testing

```python
# noise_utils.py - noise injection for ablation experiments
make_each_cam_noise(cam_num, t_noise_bound, r_noise_bound):
  For each camera:
    translation = random_unit_vector() × t_noise_bound × rand()
    rotation = axis_angle(random_axis, r_noise_bound × rand())
    → 4×4 noise transform matrix

# Usage: --t_noise_bound 0.1 --r_noise_bound 5
# Meaning: max 0.1m translation noise + max 5° rotation noise
# Note: automatically disabled when use_rig && from_lidar (L467: t_noise, r_noise = 0.0, 0.0)
```

### 13.6 Voxel Prefiltering Acceleration

```
┌──────────────────────────────────────────────────────────────────────┐
│                  Voxel Prefiltering Acceleration (View Frustum Culling)│
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  prefilter_voxel() executed before each render:                       │
│                                                                      │
│  ① Take all anchor points (N points)                                  │
│  ② Use rasterizer visible_filter to quickly detect anchors in frustum │
│  ③ Return visibility mask: [N] bool                                   │
│  ④ Generate auxiliary gaussians only for visible anchors              │
│                                                                      │
│  Effect:                                                              │
│  • In large scenes, only 30-50% of anchors may be visible per frame │
│  • Skip MLP computation for invisible anchors, significantly reducing compute│
│  • visible_filter uses simplified 3D covariance projection, much faster than full render│
│                                                                      │
│  Code: gaussian_renderer/__init__.py::prefilter_voxel()                │
│  Call: voxel_visible_mask = prefilter_voxel(cam, gs, pipe, bg)        │
│        render(..., visible_mask=voxel_visible_mask)                    │
└──────────────────────────────────────────────────────────────────────┘
```

### 13.7 Extrinsic Generalization Capability

**Important practical property demonstrated in Paper Table VII**:

```
┌──────────────────────────────────────────────────────────────────────┐
│                   Extrinsic Cross-Scene Generalization Experiment    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Experiment design:                                                  │
│  • Optimize extrinsics T^e_c on Scene A                              │
│  • Apply T^e_c directly to Scene B, C (same vehicle collection)      │
│  • Measure extrinsic generalization via NVS quality                    │
│                                                                      │
│  Results:                                                            │
│  Extrinsics optimized on Scene A → Scene B NVS quality > dataset original calibration│
│                                                                      │
│  Significance:                                                       │
│  • Proves TLC-Calib estimates true sensor intrinsic parameters         │
│  • Not overfitting to a specific scene                                 │
│  • One calibration applicable to all scenes from the same vehicle    │
│  • Rig optimization is key to this property                            │
│    (per-image optimization would scene-overfit)                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 13.8 Depth Rendering and Gradient Propagation

TLC-Calib adds **depth map rendering** and corresponding pose gradient paths on top of standard 3DGS:

```
Forward: depth(u) = Σ_i d_i · α_i · G^{2D}_i(u) · Π_{j<i}(1 - α_j · G^{2D}_j(u))
      where d_i = p_C.z (Gaussian center z-coordinate in camera frame)

Backward (backward.cu::preprocessCUDA):
  dL_dpCz = dL_ddepth[idx]
  // Depth gradient w.r.t. 3D mean
  dL_dmeans[idx].x += dL_dpCz * viewmatrix[2]
  dL_dmeans[idx].y += dL_dpCz * viewmatrix[6]
  dL_dmeans[idx].z += dL_dpCz * viewmatrix[10]
  
  // Depth gradient w.r.t. pose (Path ④)
  for i in 0..2:
    dL_dtau[6*idx + i]   += dL_dpCz * dp_C_d_rho.cols[i].z
    dL_dtau[6*idx + i+3] += dL_dpCz * dp_C_d_theta.cols[i].z
```

Although the depth gradient path is not explicitly used for depth supervision in the current loss function, it indirectly affects pose gradients through alpha compositing backpropagation, providing additional geometric constraints for pose optimization.

---

## 14. Deep Analysis of the CUDA Forward Rendering Pipeline

> **Source path**: `submodules/diff-gaussian-rasterization-w-pose/cuda_rasterizer/forward.cu`  
> **Related files**: `rasterizer_impl.cu` (scheduling layer), `auxiliary.h` (auxiliary functions), `config.h` (constants), `math.h` (Lie group math)

### 14.1 Tile-Based Rasterization Overall Architecture

TLC-Calib's forward rendering pipeline strictly follows the original 3DGS Tile-Based Rasterization architecture, injecting pose parameters (`theta`, `rho`) at key positions to support differentiable calibration. The overall flow consists of three core stages:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    CUDA Tile-Based Forward Rendering Pipeline                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Stage 1: preprocessCUDA (per Gaussian, 256 threads/block)                 │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │  World Space 3D points ──→ Frustum culling ──→ Project to NDC     │     │
│  │  ──→ Compute 3D covariance ──→ Project to 2D covariance ──→ Inverse to Conic│
│  │  ──→ Compute eigenvalues ──→ Bounding box (tile coverage) ──→ Record depth│
│  │  ──→ Optional: SH→RGB (TLC-Calib uses MLP precomputed, skip this step)│  │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                              ↓                                               │
│  Stage 2: Sorting and Tile allocation (CPU + GPU combined)                   │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │  tiles_touched[] ──→ InclusiveSum (prefix sum)                     │     │
│  │  ──→ duplicateWithKeys: generate [tile_id | depth] key-value pairs │     │
│  │  ──→ RadixSort (sort by key) ──→ identifyTileRanges                │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                              ↓                                               │
│  Stage 3: renderCUDA (per tile, 16×16=256 threads/block)                   │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │  Thread block cooperatively loads Gaussians ──→ Per-pixel alpha-blending│ │
│  │  ──→ Color + depth + opacity ──→ Background blend ──→ Write framebuffer│ │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Configuration constants** (`config.h`):
- `BLOCK_X = 16`, `BLOCK_Y = 16` → each tile covers 16×16 = 256 pixels
- `NUM_CHANNELS = 3` → RGB three channels
- `BLOCK_SIZE = 256` → each thread block processes one tile

### 14.2 preprocessCUDA: Per-Gaussian Preprocessing Kernel

This is the most computationally intensive stage of the forward pipeline; each thread processes one Gaussian.

**Step 1: Frustum culling**
```c
// auxiliary.h: in_frustum()
float3 p_view = transformPoint4x3(p_orig, viewmatrix);
if (p_view.z <= 0.2f)  // near plane culling threshold
    return false;
```
- Transform 3D point to camera coordinates, check if behind near plane (z > 0.2)
- Failed points marked `radii[idx] = 0`, all subsequent steps skipped

**Step 2: Perspective projection to NDC**
```c
float3 p_orig = {orig_points[3*idx], orig_points[3*idx+1], orig_points[3*idx+2]};
float4 p_hom = transformPoint4x4(p_orig, projmatrix);  // homogeneous coordinates
float p_w = 1.0f / (p_hom.w + 0.0000001f);            // avoid division by zero
float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};  // NDC
```
- `projmatrix` = `viewmatrix * projection_matrix`, pre-multiplied on Python side
- Note: `projmatrix` here already includes current frame pose increment delta (applied via `update_pred_pose` on Python side)

**Step 3: 3D covariance matrix computation** (`computeCov3D`)
```c
// From scale + quaternion → 3D covariance matrix
glm::mat3 S = diag(mod * scale.x, mod * scale.y, mod * scale.z);
glm::mat3 R = quaternion_to_matrix(rot);  // quaternion→rotation matrix
glm::mat3 M = S * R;
glm::mat3 Sigma = transpose(M) * M;      // positive semi-definite guarantee
// Store upper triangle only (6 elements): Sigma[0:5]
```

**Step 4: 3D→2D covariance projection** (`computeCov2D`)
```
Mathematical derivation (EWA Splatting, Zwicker et al., 2002):

Given:
  - 3D covariance Σ_3D (world coordinates)
  - View matrix W (world → camera)
  - Projection Jacobian J (camera → image)

Projection:
  T = W × J
  Σ_2D = T^T × Σ_3D × T

Implementation:
  (1) Transform to camera coords: t = W × p_world
  (2) Compute Jacobian:
      J = | fx/tz    0    -fx*tx/tz²  |
          |   0    fy/tz  -fy*ty/tz²  |
          |   0      0        0       |
  (3) Combine: T = W × J
  (4) Project: cov2D = T^T × Vrk × T
  (5) Low-pass filter: cov2D[0][0] += 0.3; cov2D[1][1] += 0.3
```
- **Field of view limit** (`limx = 1.3 * tan_fovx`): clamp points outside FOV to prevent extreme projections
- **Low-pass filter** (+0.3): ensures each Gaussian covers at least ~1 pixel, avoiding sub-pixel aliasing

**Step 5: Conic (inverse covariance) computation**
```c
float det = cov.x * cov.z - cov.y * cov.y;  // determinant
float3 conic = {cov.z/det, -cov.y/det, cov.x/det};  // 2×2 symmetric matrix inverse
```
- When determinant is zero the entire Gaussian is skipped (degenerate ellipse)

**Step 6: Screen coverage range computation**
```c
// Eigenvalues → 3σ radius
float mid = 0.5f * (cov.x + cov.z);
float lambda1 = mid + sqrt(max(0.1f, mid*mid - det));
float lambda2 = mid - sqrt(max(0.1f, mid*mid - det));
float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

// NDC → pixel coordinates
float2 point_image = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};
// ndc2Pix: ((v + 1.0) * S - 1.0) * 0.5

// Compute covered tile rectangle range
getRect(point_image, my_radius, rect_min, rect_max, grid);
tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
```

**Step 7: Color computation branch**
```c
if (colors_precomp == nullptr) {
    // Standard 3DGS: SH → RGB (up to 4th order, 16 coefficients/channel)
    glm::vec3 result = computeColorFromSH(idx, D, M, means, campos, shs, clamped);
} else {
    // TLC-Calib: use MLP precomputed colors (Python generate_neural_gaussians)
    // colors_precomp already contains RGB, skip SH
}
```

**Key outputs**:
| Output | Size/Gaussian | Purpose |
|------|---------------|------|
| `depths[idx]` | float | Front-to-back sorting |
| `radii[idx]` | int | Visibility determination |
| `points_xy_image[idx]` | float2 | Pixel coordinates |
| `conic_opacity[idx]` | float4 | Conic 3 + opacity 1 packed |
| `tiles_touched[idx]` | uint32 | Number of covered tiles |

### 14.3 renderCUDA: Per-Pixel Alpha-Blending Kernel

This is the final pixel shading stage; each thread processes one pixel.

```
Thread block organization:
  - grid:   (ceil(W/16), ceil(H/16))  → each block corresponds to one tile
  - block:  (16, 16)                  → each thread corresponds to one pixel

Shared memory (per tile):
  - collected_id[256]          → Gaussian global ID
  - collected_xy[256]          → 2D projected coordinates
  - collected_conic_opacity[256] → Conic + opacity
  - collected_depth[256]       → Depth values
```

**Alpha-blending core loop**:
```c
float T = 1.0f;        // accumulated transmittance, initial 1 (fully transparent)
float C[3] = {0};      // accumulated color
float D = 0.0f;        // accumulated depth

for (each batch of Gaussians in sorted order) {
    // Cooperative load: all 256 threads load from global memory to shared memory
    if (range.x + progress < range.y) {
        collected_id[tid] = point_list[range.x + progress];
        collected_xy[tid] = points_xy_image[coll_id];
        // ...
    }
    __syncthreads();  // sync wait for load completion

    for (each Gaussian j in batch) {
        // Compute Mahalanobis distance from pixel to Gaussian center
        float2 d = {xy.x - pixf.x, xy.y - pixf.y};
        float power = -0.5f * (con_o.x*d.x*d.x + con_o.z*d.y*d.y) - con_o.y*d.x*d.y;
        if (power > 0.0f) continue;  // outside Gaussian

        // Alpha = opacity × Gaussian falloff
        float alpha = min(0.99f, con_o.w * exp(power));
        if (alpha < 1.0f/255.0f) continue;  // contribution too small

        // Early termination check
        float test_T = T * (1 - alpha);
        if (test_T < 0.0001f) { done = true; continue; }

        // Alpha-compositing (front-to-back)
        for (int ch = 0; ch < 3; ch++)
            C[ch] += features[id * 3 + ch] * alpha * T;
        D += depth[j] * alpha * T;  // depth also alpha-blended

        // Count pixels covered (for densification)
        if (test_T > 0.5f)
            atomicAdd(&n_touched[id], 1);

        T = test_T;
    }
}

// Final: add background color
for (int ch = 0; ch < 3; ch++)
    out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
out_depth[pix_id] = D;
out_opacity[pix_id] = 1 - T;
```

**Key design details**:

1. **Front-to-back sorted rendering**: Sort key is `[tile_id | depth]`, ensuring within-tile traversal from near to far
2. **Early stop**: Stop blending when `T < 0.0001`, reducing invalid computation
3. **Whole block vote**: `__syncthreads_count(done)` — skip entire batch only when all threads complete
4. **n_touched statistics**: Count only when `T > 0.5`, used for subsequent anchor growing/pruning densification criteria
5. **Depth rendering**: Alpha-blending in parallel with color, providing depth gradient path for backpropagation

### 14.4 Sorting and Per-Tile Scheduling

Sorting stage orchestrated by `Rasterizer::forward()` in `rasterizer_impl.cu`:

```
Step 1: InclusiveSum (prefix sum)
  tiles_touched = [2, 3, 0, 2, 1]
  point_offsets = [2, 5, 5, 7, 8]  ← start offset for each Gaussian write

Step 2: duplicateWithKeys
  For each tile covered by each Gaussian, generate:
    key = (tile_y * grid.x + tile_x) << 32 | depth_as_uint32
    value = gaussian_idx

Step 3: RadixSort (CUB library)
  Sort by key → within same tile sorted by depth

Step 4: identifyTileRanges
  Scan sorted keys, mark [start, end) range for each tile
  ranges[tile_id] = {start_idx, end_idx}
```

**Memory layout — three state buffer types**:
| Buffer | Lifetime | Content |
|---------|----------|------|
| `GeometryState` | Per Gaussian | depths, means2D, cov3D, conic_opacity, rgb, tiles_touched |
| `BinningState` | Sorting intermediate | key/value pairs (unsorted+sorted), sorting workspace |
| `ImageState` | Per pixel | accum_alpha (final_T), n_contrib, tile ranges |

### 14.5 filter_preprocessCUDA: Voxel Prefiltering Dedicated Kernel

Lightweight preprocessing kernel for TLC-Calib's `prefilter_voxel()`:

```c
// Compared to preprocessCUDA:
// ✓ Frustum culling
// ✓ 3D→2D covariance projection
// ✓ Eigenvalues → radius computation
// ✗ No color computation (no SH/color_precomp)
// ✗ No depth computation
// ✗ No tiles_touched computation
// ✗ No conic_opacity storage
```

The only output is `radii[idx]`; non-zero means visible. Python side generates `visible_mask` accordingly:
```python
# gaussian_renderer/__init__.py
radii_pure = rasterizer.visible_filter(means3D=means3D, scales=scales[:,:3], ...)
return radii_pure > 0  # → bool tensor, passed to generate_neural_gaussians
```

This enables Neural Gaussian generation only on visible anchors, saving 30-50% computation.

### 14.6 Symmetric Design of Forward and Backward Passes

```
Forward (forward.cu)                          Backward (backward.cu)
┌──────────────────────────┐               ┌──────────────────────────┐
│ preprocessCUDA           │               │ BACKWARD::render         │
│   computeCov3D           │  ←──symmetric──→│   renderCUDA (backward)  │
│   computeCov2D           │               │ BACKWARD::preprocess     │
│   computeColorFromSH     │               │   computeCov2DCUDA       │
│ renderCUDA               │               │   preprocessCUDA (backward)│
│   alpha-blending         │               │     computeColorFromSH   │
│   depth-blending         │               │     computeCov3D         │
└──────────────────────────┘               └──────────────────────────┘

Key differences:
  Backward renderCUDA: traverses in reverse order
  Backward preprocessCUDA: additionally computes dL/dtau (6-DOF pose gradient)
  Backward computeCov2DCUDA: additionally computes dL/d(W) → dL/d(theta) gradient
```

Each forward computation step has a corresponding analytic gradient implementation in backward — no PyTorch autograd dependency; all hand-written in CUDA. This is one of the core reasons TLC-Calib can train efficiently.

---

## 15. render.py Visualization Rendering Script Analysis

> **Source path**: `render.py`  
> **Role**: Offline visualization rendering and quality assessment tool after training

### 15.1 Script Purpose and Role

`render.py` is TLC-Calib's **offline rendering and visualization tool**, used for visual assessment of calibration results after training completes. It does not participate in training or NVS evaluation (that is `metrics_nvs.py`'s job):

```
┌─────────────────────────────────────────────────────────────┐
│                    render.py Workflow                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input                                                       │
│  ├── Trained model (-m model_path)                            │
│  ├── Specified iteration (--iteration, default latest)        │
│  └── Dataset parameters (auto-loaded from cfg_args)           │
│                                                             │
│  Processing pipeline                                          │
│  ├── Load GaussianModel + Scene (with calibrated poses)       │
│  ├── Set eval mode (gaussians.eval())                         │
│  ├── For each viewpoint:                                      │
│  │   ├── update_pred_pose() → apply calibrated delta pose     │
│  │   ├── prefilter_voxel() → frustum culling                  │
│  │   ├── render() → forward rendering                         │
│  │   ├── Timing (FPS statistics)                              │
│  │   ├── Compute error map = |render - gt|                    │
│  │   └── Save render, gt, error, compare images               │
│  └── Save per_view_count.json (visible Gaussian count per view)│
│                                                             │
│  Output                                                       │
│  ├── renders/     → pure rendering results                    │
│  ├── gt/          → Ground Truth images                       │
│  ├── errors/      → absolute error maps (diagnose calibration)│
│  ├── compares/    → left-right render vs GT comparison        │
│  └── per_view_count.json → visibility statistics              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 15.2 render_set Function Details

Key steps of the core rendering function `render_set()`:

```python
def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    # 1. Create four output directories
    compare_path = .../compares
    render_path  = .../renders
    error_path   = .../errors
    gts_path     = .../gt

    for idx, view in enumerate(views):
        # 2. Apply calibrated pose (critical!)
        update_pred_pose(view, gaussians)

        # 3. Rendering pipeline (identical to train.py)
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background,
                          visible_mask=voxel_visible_mask)

        # 4. Extract rendering results
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()

        # 5. Compute absolute error map
        gt = view.original_image[0:3, :, :]
        errormap = (rendering - gt).abs()

        # 6. Save four visualization outputs
        save_image(rendering, render_path/...)
        save_image(errormap,  error_path/...)
        save_image(gt,        gts_path/...)
        save_image(cat([rendering, gt], dim=-1), compare_path/...)  # side-by-side
```

**Special notes**: 
- `update_pred_pose(view, gaussians)` is called before rendering — ensures rendering uses **calibrated poses** not initial poses
- `gaussians.eval()` disables dropout/batchnorm; TLC-Calib MLPs don't use these layers; mainly ensures inference consistency
- Timing skips first 5 frames (`t_list[5:]`) to exclude GPU warmup overhead

### 15.3 Output Directory Structure

```
output/model_path/
├── train/ours_30000/         # Training set rendering (skipped by default)
│   ├── renders/              # 00000.png, 00001.png, ...
│   ├── gt/                   # Ground Truth
│   ├── errors/               # |render - gt| absolute error heatmaps
│   ├── compares/             # [render | gt] side-by-side comparison
│   └── per_view_count.json   # {"00000.png": 12345, ...}
└── test/ours_30000/          # Test set rendering
    ├── renders/
    ├── gt/
    ├── errors/
    ├── compares/
    └── per_view_count.json
```

**Diagnostic value of error maps**:

Error maps visually demonstrate calibration quality:
- **Sharp edge errors** → calibration still has bias, projection misaligned
- **Uniform low error** → precise calibration, good scene reconstruction
- **High error in sky/distance** → LiDAR blind zones, auxiliary Gaussians incomplete coverage
- **High error on moving objects** → dynamic scene, not a calibration issue

### 15.4 Relationship with train.py and metrics_nvs.py

```
┌────────────────────────────────────────────────────────────┐
│                      Division of Three Scripts              │
├──────────┬──────────┬─────────────┬────────────────────────┤
│          │ train.py │ render.py   │ metrics_nvs.py         │
├──────────┼──────────┼─────────────┼────────────────────────┤
│ Purpose  │ Calib train│ Visual render│ NVS quality eval      │
│ Pose opt │ ✓        │ ✗           │ ✗                      │
│ Scene opt│ ✓        │ ✗           │ ✓ (independent 3DGS retrain)│
│ Renderer │ w-pose   │ w-pose      │ Standard 3DGS          │
│ Output   │ Model weights│ Images+errors│ PSNR/SSIM/LPIPS     │
│ Rig opt  │ ✓        │ ✗           │ ✗                      │
│ Metrics  │ Convergence│ FPS/visual  │ Quantitative NVS metrics│
└──────────┴──────────┴─────────────┴────────────────────────┘
```

The unique value of `render.py` is providing **manual visual inspection** capability — error maps and comparison images are important material for demonstrating calibration results in paper figures.

---

## 16. TLC-Calib vs. Latest 3DGS Calibration Methods

### 16.1 Method Lineage Diagram

```
Timeline (2023 → 2026):

NeRF Era
┌──────────────────────────────────────────────────────────────────┐
│  iNeRF (2021)  →  MOISST (2023)  →  INF (2024)                 │
│    ↓                 ↓                  ↓                        │
│  Single-frame NeRF  NeRF+calibration  Improved NeRF              │
│  pose estimation    multi-sensor      temporal modeling          │
│                   >5h training        >4h training               │
└──────────────────────────────────────────────────────────────────┘
                              ↓ 3DGS revolution (2023)
Gaussian Splatting Era
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  3DGS-Calib (2024)        CalibAnything (2024)                  │
│    ↓                        ↓                                    │
│  First 3DGS calibration   Zero-training, foundation models      │
│  Hash-Grid encoding       Feature matching + PnP                 │
│  ~9min training           >5h (feature extraction)               │
│                                                                  │
│  RobustCalib (2025)       HiGS-Calib (2025)                    │
│    ↓                        ↓                                    │
│  2DGS geometry recon.     Hierarchical coarse-to-fine            │
│  Reprojection+triangulation LCPG error model                     │
│  >1h training             Decoupled modeling and calibration     │
│                                                                  │
│  ★ TLC-Calib (2026) ★                                           │
│    ↓                                                             │
│  Neural Gaussian (Anchor + Auxiliary)                            │
│  CUDA analytic pose gradients                                    │
│  Rig optimization + AVC + Refine                                 │
│  ~11min training                                                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 16.2 Per-Method Deep Comparison

#### A. 3DGS-Calib (Hérau et al., ICRA 2024)

**Core approach**: Fix LiDAR points as 3DGS centers, use hash-grid encoded MLP to generate Gaussian attributes, jointly optimize pose and scene.

**Key differences vs. TLC-Calib**:

| Dimension | 3DGS-Calib | TLC-Calib |
|------|-----------|-----------|
| Point position | Fixed on LiDAR points | Anchor fixed + Auxiliary learnable |
| Attribute generation | Hash-Grid MLP | View-adaptive MLP |
| Coverage range | LiDAR-covered regions only | Extended to LiDAR blind zones |
| Density control | None | AVC (Adaptive Voxel Control) |
| Spatiotemporal calibration | ✓ (supports time offset) | ✓ (supports time offset) |
| Hyperparameter sensitivity | High (hash-grid encoding) | Low (anchor+MLP architecture) |

**TLC-Calib paper analysis of 3DGS-Calib**: Binding all Gaussians to LiDAR points leads to:
1. **LiDAR blind zones cannot be rendered** → sky, distant buildings lack supervision
2. **Hash-grid encoding sensitive to scene complexity** → different scenes need tuning
3. **No auxiliary Gaussians** → easily overfits wrong pose under noisy initialization

#### B. RobustCalib (Zhou et al., RA-L 2025)

**Core approach**: Two-stage strategy — first learn geometry from LiDAR point cloud via 2D Gaussian Splatting (2DGS), then calibrate extrinsics via reprojection and triangulation losses.

**Key differences vs. TLC-Calib**:

| Dimension | RobustCalib | TLC-Calib |
|------|------------|-----------|
| GS type | 2DGS (planar) | 3DGS (volumetric) |
| Optimization strategy | Two-stage decoupled | Joint optimization + Refine |
| Loss functions | Photometric + reprojection + triangulation | Photometric (L1+D-SSIM) + scale regularization |
| Normal dependency | ✓ (2DGS needs normals) | ✗ |
| Rig optimization | ✗ (per-frame) | ✓ (shared extrinsics) |
| Sparse LiDAR | Degrades (poor normal estimation) | Robust (no normal dependency) |

**TLC-Calib advantage**: RobustCalib depends on 2DGS surface normal quality; performance degrades under sparse LiDAR (e.g., solid-state LiDAR). TLC-Calib's Neural Gaussian architecture does not depend on normal estimation and performs well on Fast-LIVO2 (solid-state LiDAR) data.

#### C. HiGS-Calib (Zhang et al., TCSVT 2025)

**Core approach**: Proposes LCPG (Local-Consistent Photometric-Geometric) error model, using spatial consistency within local windows to quantify pose deviation, with hierarchical coarse-to-fine iterative optimization.

**Key differences vs. TLC-Calib**:

| Dimension | HiGS-Calib | TLC-Calib |
|------|-----------|-----------|
| Optimization paradigm | Decoupled (model first, then calibrate) | Joint (scene+pose simultaneous) |
| Error model | LCPG (geometric attributes) | Pure photometric + regularization |
| Color dependency | Circumvents color-pose coupling | Directly uses color gradients |
| Hierarchy | Coarse→fine iteration | Single stage + Refine |
| External dependencies | ROMA optical flow model | None (end-to-end) |
| LiDAR error tolerance | ✓ (local consistency) | ✓ (anchor freezing for stability) |

**HiGS-Calib unique contribution**: Solves the "chicken and egg" dilemma in joint optimization (precise calibration needs precise scene, precise scene needs precise calibration) via decoupling strategy to avoid oscillation.

**TLC-Calib response**: Through anchor freezing + Warmup + Rig sharing + Refine stage, solves the same problem within joint optimization framework without external optical flow model dependency.

#### D. CalibAnything (Luo et al., 2024)

**Core approach**: Zero-training method using foundation models (SAM, DINOv2) to extract semantic features from LiDAR and images, achieving calibration via feature matching.

**Key differences vs. TLC-Calib**:

| Dimension | CalibAnything | TLC-Calib |
|------|-------------|-----------|
| Training requirement | Zero-training | Requires optimization (~11min) |
| Representation | Feature matching | Neural Gaussian |
| Accuracy | Moderate | High |
| Generalization | Good cross-domain | Best in-domain |
| Speed | >5h (feature extraction) | ~11min |
| Multi-camera | Per-camera | Rig unified optimization |

#### E. INF (NeRF-based, 2024)

Represents the final improvements of the NeRF era; speed >4h, accuracy comprehensively surpassed by 3DGS methods.

### 16.3 Quantitative Performance Comparison

Based on Paper Table I (KITTI-360 dataset) results:

```
┌──────────────────────────────────────────────────────────────────────────┐
│              KITTI-360 Calibration Accuracy Comparison (avg R/t error)  │
├──────────────────┬──────────┬──────────┬──────────┬──────────────────────┤
│ Method           │ R (°) ↓  │ t (cm) ↓ │ SR (%) ↑ │ Training Time        │
├──────────────────┼──────────┼──────────┼──────────┼──────────────────────┤
│ CalibAnything     │  ~1.5    │  ~5.0    │  ~50     │ > 5h                │
│ INF (NeRF)       │  ~0.5    │  ~2.0    │  ~75     │ > 4h                │
│ RobustCalib       │  ~0.3    │  ~1.5    │  ~85     │ > 1h                │
│ 3DGS-Calib       │  ~0.4    │  ~9.6    │  ~80     │ ~9 min (0.15h)      │
│ ★ TLC-Calib ★    │  ~0.13   │  ~0.8    │  ~95     │ ~11 min (0.18h)    │
└──────────────────┴──────────┴──────────┴──────────┴──────────────────────┘

Note: SR = Success Rate (proportion with rotation<1° and translation<5cm)
Source: TLC-Calib Paper Table I, values are approximate averages
```

### 16.4 Technical Approach Comparison Table

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Technical Approach Panorama Comparison                     │
├──────────────┬──────────┬────────────┬──────────┬──────────┬──────────────────┤
│ Feature      │3DGS-Calib│RobustCalib │HiGS-Calib│CalibAnyt.│ ★ TLC-Calib ★    │
├──────────────┼──────────┼────────────┼──────────┼──────────┼──────────────────┤
│ Scene repr.  │ 3DGS     │ 2DGS       │ 3DGS     │ Features │ Neural GS        │
│              │+HashGrid │            │          │          │(Anchor+Auxiliary)│
│ Joint opt.   │ ✓        │ ✗ (2-stage)│ ✗ (decoup)│ ✗        │ ✓                │
│ Rig opt.     │ ✗        │ ✗          │ ✗        │ ✗        │ ✓                │
│ LiDAR blind  │ ✗        │ ✗          │ ✗        │ N/A      │ ✓ (aux Gaussians)│
│ Adaptive dens│ ✗        │ ✗          │ ✗        │ N/A      │ ✓ (AVC)          │
│ CUDA pose grad│ ✗        │ ✗          │ ✗        │ N/A      │ ✓ (analytic grad)│
│ Refine stage │ ✗        │ ✗          │ Coarse→fine│ ✗        │ ✓ (frozen pose)  │
│ External model│ ✗        │ ✗          │ ✓ (ROMA) │ ✓ (SAM)  │ ✗                │
│ Solid-state  │ ✗        │ Degrades   │ ?        │ ?        │ ✓ (FAST-LIVO2)   │
│ LiDAR        │          │            │          │          │                  │
│ Time offset  │ ✓        │ ✗          │ ✗        │ ✗        │ ✓                │
│ NVS eval     │ ✗        │ ✗          │ ✗        │ ✗        │ ✓ (independent pipeline)│
└──────────────┴──────────┴────────────┴──────────┴──────────┴──────────────────┘
```

### 16.5 TLC-Calib Differentiating Advantages

**1. Only end-to-end joint optimization + Rig constraint**

TLC-Calib is the only method implementing Rig optimization within the 3DGS framework. All cameras share extrinsics; each update affects all training frames, providing strong regularization. Ablation shows removing Rig degrades rotation error 15×.

**2. Most complete CUDA-level pose gradients**

Four independent gradient paths (2D mean, 2D cov, depth, SH/color) implemented analytically, more efficient than automatic differentiation. Other methods mostly rely on PyTorch autograd or finite differences.

**3. LiDAR blind zone extension capability**

Auxiliary Gaussians are a TLC-Calib unique design, enabling photometric gradients from sky, distant buildings, and other LiDAR-unreachable regions for full-image supervision.

**4. Broadest sensor coverage**

Only method validated on three different datasets (KITTI-360 spinning LiDAR + Waymo spinning LiDAR + Fast-LIVO2 solid-state LiDAR), demonstrating cross-sensor modality generalization.

**5. Complete evaluation ecosystem**

Provides NVS evaluation pipeline (`metrics_nvs.py`), pose evaluation (`metrics_pose.py`), batch evaluation (`full_eval.py`), and real-time Web Viewer — the most complete open-source evaluation implementation in 3DGS calibration.

---

## 17. Deep Analysis of CUDA Backward Rendering Kernels

> **Source path**: `submodules/diff-gaussian-rasterization-w-pose/cuda_rasterizer/backward.cu`  
> **Focus**: Reverse traversal in backward `renderCUDA`, gradient decomposition, four-path convergence of pose gradients

### 17.1 Reverse Traversal Mechanism in Backward renderCUDA

Forward rendering blends front-to-back; backward rendering traverses **back-to-front**. This is required by the chain rule of alpha-compositing for reverse-order access.

```
Forward rendering (front-to-back):       Backward rendering (back-to-front):
  range.x ──→ range.y                     range.y ──→ range.x
  G₁ → G₂ → G₃ → ... → Gₙ               Gₙ → ... → G₃ → G₂ → G₁

Forward:                                Backward:
  T = 1.0                                T = T_final (saved from forward)
  for i = 0 → N:                         for i = N → 0:
    C += color[i] * α[i] * T               ∂L/∂α[i], ∂L/∂color[i]
    T *= (1 - α[i])                         T /= (1 - α[i])  ← key: division recovery
```

**Core data recovery mechanism**:

```c
// In backward, recover T at each step from T_final
T = T_final;  // saved from forward

// Reverse recovery during traversal
T = skip ? T : T / (1.f - alpha);  // division recovers previous T
```

**Shared memory structure** (symmetric to forward, but adds gradient accumulation buffers):

```c
// Data reused from forward
__shared__ int collected_id[BLOCK_SIZE];
__shared__ float2 collected_xy[BLOCK_SIZE];
__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
__shared__ float collected_colors[C * BLOCK_SIZE];
__shared__ float collected_depths[BLOCK_SIZE];

// New in backward: gradient accumulation shared memory (for warp-level reduce)
__shared__ float2 dL_dmean2D_shared[BLOCK_SIZE];
__shared__ float3 dL_dcolors_shared[BLOCK_SIZE];
__shared__ float dL_ddepths_shared[BLOCK_SIZE];
__shared__ float dL_dopacity_shared[BLOCK_SIZE];
__shared__ float4 dL_dconic2D_shared[BLOCK_SIZE];
```

### 17.2 Per-Pixel Gradient Decomposition

For each pixel-Gaussian pair, backpropagation computes the following gradients:

**1. Color gradient** `dL/d(color[i])`:
```c
// dchannel_dcolor = α[i] × T
const float dchannel_dcolor = alpha * T;
for (int ch = 0; ch < C; ch++) {
    local_dL_dcolors[ch] = skip ? 0.0f : dchannel_dcolor * dL_dpixel[ch];
}
```

**2. Alpha gradient** `dL/d(α[i])`:
```c
// Color part: ∂L/∂α = Σ_ch (c[i] - R[i]) × ∂L/∂pixel[ch] × T
float dL_dalpha = 0.0f;
for (int ch = 0; ch < C; ch++) {
    accum_rec[ch] = last_alpha * last_color[ch] + (1 - last_alpha) * accum_rec[ch];
    dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
}

// Depth part: same alpha-compositing chain rule
dL_dalpha += (depth - accum_rec_depth) * dL_dpixel_depth;

// Multiply by T
dL_dalpha *= T;

// Background color correction: account for alpha's effect on background contribution
float bg_dot_dpixel = Σ(bg_color[i] * dL_dpixel[i]);
dL_dalpha += (-T_final / (1 - alpha)) * bg_dot_dpixel;
```

**3. 2D mean gradient** `dL/d(mean2D)`:
```c
const float dL_dG = con_o.w * dL_dalpha;  // ∂L/∂G = opacity × ∂L/∂α
const float gdx = G * d.x;
const float gdy = G * d.y;
const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;  // ∂G/∂Δx
const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;  // ∂G/∂Δy

dL_dmean2D.x = dL_dG * dG_ddelx * (0.5f * W);  // multiply NDC→pixel Jacobian
dL_dmean2D.y = dL_dG * dG_ddely * (0.5f * H);
```

**4. Conic gradient** `dL/d(conic)`:
```c
dL_dconic2D.x = -0.5f * gdx * d.x * dL_dG;  // ∂L/∂conic_xx
dL_dconic2D.y = -0.5f * gdx * d.y * dL_dG;  // ∂L/∂conic_xy
dL_dconic2D.w = -0.5f * gdy * d.y * dL_dG;  // ∂L/∂conic_yy
```

### 17.3 Warp-Level Reduce Sum Optimization

TLC-Calib's backward rendering uses a **key optimization** — `render_cuda_reduce_sum`:

```c
template <typename group_t, typename... Lists>
__device__ void render_cuda_reduce_sum(group_t g, Lists... lists) {
    int lane = g.thread_rank();
    g.sync();
    for (int i = g.size() / 2; i > 0; i /= 2) {
        (..., reduce_helper(lane, i, lists));  // C++17 fold expression
        g.sync();
    }
}
```

```
256 threads process gradients for the same Gaussian in one tile:

Thread[0..255] each compute dL/d(mean2D), dL/d(conic), dL/d(opacity), dL/d(color)

Reduce Sum (log₂(256) = 8 steps):
  Step 1: threads [0..127] += [128..255]
  Step 2: threads [0..63]  += [64..127]
  ...
  Step 8: thread [0]      += [1]

Thread[0] performs atomicAdd to global memory

Advantage: reduces 256 atomicAdds to 1 → 8× reduction in atomic contention
```

**Skip Counter optimization**: When all 256 threads in the block skip a Gaussian, directly `continue` to avoid invalid reduce operations.

```c
if (skip) atomicAdd(&skip_counter, 1);
block.sync();
if (skip_counter == BLOCK_SIZE) continue;  // all skipped
```

### 17.4 Pose Gradient Injection in computeCov2DCUDA

This is the **second path** of pose gradients (2D covariance → pose). Gradients flow from `dL/d(conic)` through inverse covariance, 2D covariance, projection Jacobian T = W×J, ultimately reaching pose.

**Pose gradients propagate to t (camera coordinates) via Jacobian:**

```c
SE3 T_CW(view_matrix);
mat33 R = T_CW.R().data();
float3 t = transformPoint4x3(mean, view_matrix);  // recompute camera coordinates

// ∂(p_cam)/∂(ρ) = I (translation Jacobian w.r.t. translation)
mat33 dpC_drho = mat33::identity();
// ∂(p_cam)/∂(θ) = -[p_cam]× (rotation Lie algebra Jacobian)
mat33 dpC_dtheta = -mat33::skew_symmetric(t);

// Chain rule: dL/dτ = dL/dt × dt/dτ
float dL_dt[6];
for (int i = 0; i < 3; i++) {
    float3 c_rho = dpC_drho.cols[i];
    float3 c_theta = dpC_dtheta.cols[i];
    dL_dt[i]     = dL_dtx * c_rho.x   + dL_dty * c_rho.y   + dL_dtz * c_rho.z;
    dL_dt[i + 3] = dL_dtx * c_theta.x + dL_dty * c_theta.y + dL_dtz * c_theta.z;
}
// Accumulate to pose gradient
for (int i = 0; i < 6; i++)
    dL_dtau[6 * idx + i] += dL_dt[i];
```

**Rotation matrix W gradient → θ gradient:**

```c
// W = R (rotation part of view matrix)
// dW/dθ = -[R_col]× (skew-symmetric matrix)

mat33 n_W1_x = -mat33::skew_symmetric(c1);  // c1 = column 1 of R
mat33 n_W2_x = -mat33::skew_symmetric(c2);
mat33 n_W3_x = -mat33::skew_symmetric(c3);

float3 dL_dtheta = {};
dL_dtheta.x = dot(dL_dWc1, n_W1_x.cols[0]) + dot(dL_dWc2, n_W2_x.cols[0])
            + dot(dL_dWc3, n_W3_x.cols[0]);
// ... y, z similarly

dL_dtau[6*idx + 3] += dL_dtheta.x;
dL_dtau[6*idx + 4] += dL_dtheta.y;
dL_dtau[6*idx + 5] += dL_dtheta.z;
```

### 17.5 Backward preprocessCUDA: Four-Path Convergence of Pose Gradients

Backward `preprocessCUDA` is the convergence point for four pose gradient paths. `dL_dtau` accumulates via `+=` from three independent sources:

```
Sources of dL_dtau[6*idx + 0..5] accumulation:

┌─────────────────────────────────────────────────────────────────────┐
│ Path 1: 2D projected mean (inside preprocessCUDA)                   │
│   dL/dmean2D → dL/d(proj) → dL/d(p_cam) → dL/dτ                   │
│   Math: dmean2D_dtau[6] = ∂proj/∂p_cam × ∂p_cam/∂τ                │
│   Code: dL_dt[i] = dL_dmean2D.x * dmean2D_dtau[i].x              │
│                   + dL_dmean2D.y * dmean2D_dtau[i].y              │
│   Inject: dL_dtau[6*idx + i] += dL_dt[i]                            │
├─────────────────────────────────────────────────────────────────────┤
│ Path 2: Depth (inside preprocessCUDA)                               │
│   dL/ddepth → dL/d(p_cam.z) → dL/dτ                               │
│   Math: depth = p_cam.z, ∂depth/∂ρ = [0,0,1]^T                   │
│                            ∂depth/∂θ = -[p_cam]×.row(2)          │
│   Code: dL_dtau[6*idx+i]   += dL_dpCz * c_rho.z                  │
│         dL_dtau[6*idx+i+3] += dL_dpCz * c_theta.z                │
├─────────────────────────────────────────────────────────────────────┤
│ Path 3: 2D covariance (computeCov2DCUDA, independent kernel)       │
│   dL/dconic → dL/dcov2D → dL/dT → dL/dJ → dL/d(t) → dL/dτ        │
│                                   → dL/dW → dL/d(θ)               │
│   Detailed in Section 17.4                                          │
├─────────────────────────────────────────────────────────────────────┤
│ Path 4: View-dependent color (computeColorFromSH, conditionally active)│
│   dL/dcolor → dL/d(dir) → dL/d(mean) → dL/dτ                       │
│   TLC-Calib uses MLP colors; this path active only in SH mode       │
│   Code: dL_dtau[6*idx+0] += -dL_dmean.x                            │
│         dL_dtau[6*idx+1] += -dL_dmean.y                            │
│         dL_dtau[6*idx+2] += -dL_dmean.z                            │
└─────────────────────────────────────────────────────────────────────┘

Final: dL_dtau is a [P, 6] tensor (P = Gaussian count, 6 = pose DOF)
       Aggregated to [1, 6] on Python side, split into grad_theta[1,3] and grad_rho[1,3]
```

---

## 18. Python Binding Layer autograd Function Analysis

> **Source path**: `submodules/diff-gaussian-rasterization-w-pose/diff_gaussian_rasterization_w_pose/__init__.py`  
> **Function**: Bridge PyTorch autograd and CUDA rasterizer, enabling pose gradients to flow through standard backpropagation

### 18.1 _RasterizeGaussians forward/backward Mechanism

`_RasterizeGaussians` inherits from `torch.autograd.Function`, the standard interface for PyTorch custom operators:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  PyTorch autograd ←→ CUDA Rasterizer Bridge                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Python side (gaussian_renderer/__init__.py):                                │
│    rasterizer(means3D, means2D, colors, opacity, scales, rot,              │
│               theta=cam_rot_deltas[opt_id],                                │
│               rho=cam_trans_deltas[opt_id])                                │
│         ↓                                                                   │
│  _RasterizeGaussians.apply(means3D, means2D, sh, colors_precomp,          │
│    opacities, scales, rotations, cov3Ds_precomp, theta, rho, settings)    │
│         ↓                                                                   │
│  ┌─── forward() ───────────────────────────────────────────────┐           │
│  │  args = (bg, means3D, colors, opacities, scales, rotations, │           │
│  │          ..., viewmatrix, projmatrix, projmatrix_raw, ...)   │           │
│  │  _C.rasterize_gaussians(*args)                               │           │
│  │  → (color, radii, depth, opacity, n_touched)                 │           │
│  │                                                               │           │
│  │  ctx.save_for_backward(colors, means3D, scales, rotations,  │           │
│  │    cov3Ds, radii, sh, geomBuffer, binningBuffer, imgBuffer)  │           │
│  └──────────────────────────────────────────────────────────────┘           │
│         ↓ (loss.backward() triggers)                                       │
│  ┌─── backward() ──────────────────────────────────────────────┐           │
│  │  _C.rasterize_gaussians_backward(*args)                      │           │
│  │  → (grad_means2D, grad_colors, grad_opacities, grad_means3D,│           │
│  │     grad_cov3Ds, grad_sh, grad_scales, grad_rotations,      │           │
│  │     grad_tau)                                                 │           │
│  │                                                               │           │
│  │  ★ Key: grad_tau aggregation and splitting ★                │           │
│  │  grad_tau = sum(grad_tau.view(-1, 6), dim=0)  # [P,6]→[6]   │           │
│  │  grad_rho   = grad_tau[:3].view(1, -1)        # [1,3]       │           │
│  │  grad_theta = grad_tau[3:].view(1, -1)        # [1,3]       │           │
│  │                                                               │           │
│  │  return (grad_means3D, grad_means2D, grad_sh, grad_colors,  │           │
│  │          grad_opacities, grad_scales, grad_rotations,        │           │
│  │          grad_cov3Ds, grad_theta, grad_rho, None)            │           │
│  └──────────────────────────────────────────────────────────────┘           │
│         ↓                                                                   │
│  PyTorch autograd routes grad_theta → cam_rot_deltas.grad                   │
│                     grad_rho   → cam_trans_deltas.grad                       │
│  calib_optimizer.step() → updates cam_rot_deltas, cam_trans_deltas           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 18.2 Pose Gradient Aggregation and Splitting

This is the **critical junction** of the entire pose-differentiable calibration pipeline:

```python
# CUDA returns: grad_tau shape [P*6] (6 pose gradient components per Gaussian)
# P = number of visible Gaussians in current frame

# Step 1: reshape to [P, 6]
grad_tau = grad_tau.view(-1, 6)

# Step 2: sum over all Gaussians → [6]
grad_tau = torch.sum(grad_tau, dim=0)
# Physical meaning: superposition of all Gaussian contributions to pose gradient
# Correctly reflects: pose change affects projection of all Gaussians

# Step 3: split into rotation and translation gradients
grad_rho   = grad_tau[:3].view(1, -1)  # [1, 3] translation increment Lie algebra
grad_theta = grad_tau[3:].view(1, -1)  # [1, 3] rotation increment Lie algebra
```

**Why per-Gaussian on CUDA side?**

```
Each Gaussian is affected differently by pose change:
  - Near Gaussians: pose change → large 2D displacement → large gradient
  - Far Gaussians: pose change → small 2D displacement → small gradient
  - Edge Gaussians: clamped → zero gradient

So per-Gaussian computation in CUDA, summation in Python is correct
Gradient direction determined by weighted vote of Gaussian distribution
```

### 18.3 GaussianRasterizer Complete Call Chain

```python
class GaussianRasterizer(nn.Module):
    def forward(self, means3D, means2D, opacities,
                shs=None, colors_precomp=None,
                scales=None, rotations=None,
                cov3D_precomp=None,
                theta=None, rho=None):

        # Input validation: SH and precomp colors must be mutually exclusive
        if (shs is None and colors_precomp is None) or \
           (shs is not None and colors_precomp is not None):
            raise Exception(...)

        # Similarly: scale/rot and precomp 3D cov mutually exclusive
        ...

        # Empty tensors replace None (CUDA checks .size(0) == 0)
        if shs is None:        shs = torch.Tensor([])
        if colors_precomp is None: colors_precomp = torch.Tensor([])
        if theta is None:      theta = torch.Tensor([])  # when no pose optimization
        if rho is None:        rho = torch.Tensor([])

        return rasterize_gaussians(
            means3D, means2D, shs, colors_precomp,
            opacities, scales, rotations, cov3Ds_precomp,
            theta, rho, raster_settings)
```

**TLC-Calib specific `projmatrix_raw`**:

```python
class GaussianRasterizationSettings(NamedTuple):
    ...
    projmatrix : torch.Tensor     # view × proj (combined matrix)
    projmatrix_raw : torch.Tensor # pure projection matrix (no view transform)
    ...
```

Standard 3DGS only has `projmatrix`. TLC-Calib additionally passes `projmatrix_raw` because backward `preprocessCUDA` needs to extract `a, b, c, d, e` coefficients from the raw projection matrix for analytic pose gradients (see Section 17.5 Path 1).

### 18.4 Lightweight Path of visible_filter

```python
def visible_filter(self, means3D, scales=None, rotations=None, cov3D_precomp=None):
    with torch.no_grad():  # no gradients needed, purely for acceleration
        radii = _C.rasterize_gaussians_filter(
            means3D, scales, rotations, ...)
    return radii
```

This calls `filter_preprocessCUDA` in `forward.cu` (Section 14.5), performing only frustum test and 2D radius computation — no rendering, no color — purely returns visibility.

---

## 19. TLC-Calib vs. NeRF-Based Calibration Methods

### 19.1 Overview of NeRF-Based Calibration Methods

Before 3DGS emerged (2021-2023), NeRF-based calibration was the dominant paradigm for neural rendering calibration. Representative methods:

```
NeRF calibration method timeline:

2021  iNeRF        First NeRF for pose estimation (camera-only)
2023  INF          LiDAR-Camera fusion NeRF, separate geometry/color learning
2023  MOISST       Multi-sensor spatiotemporal calibration, single NeRF scene
2024  SOAC         Improved MOISST, sensor-specific NeRF + overlap awareness
```

### 19.2 Per-Method Comparison: INF / MOISST / SOAC

#### A. INF (Implicit Neural Fusion, IROS 2023)

**Core approach**: First optimize NeRF density field (geometry) from LiDAR scans, then learn color field from 360° camera images while optimizing LiDAR→Camera extrinsics.

| Dimension | INF | TLC-Calib |
|------|-----|-----------|
| Scene representation | NeRF (MLP implicit) | Neural Gaussian (explicit points) |
| Rendering | Ray marching | Tile-based rasterization |
| Geometry/color separation | ✓ (geometry first, color second) | ✗ (joint optimization) |
| Training time | >4h | ~11min |
| Real-time rendering | ✗ | ✓ (>100 FPS) |
| 360° camera | ✓ | Pinhole camera model |
| Multi-camera Rig | ✗ | ✓ |

**INF limitations**: Ray marching computational cost O(R×S) (R=ray count, S=sample count) far exceeds rasterization O(N×T) (N=Gaussian count, T=tile coverage).

#### B. MOISST (Multimodal Optimization of Implicit Scene for SpatioTemporal, 2023)

**Core approach**: Build a single NeRF scene for the entire sensor system, using reference sensor (typically LiDAR) trajectory as baseline, jointly optimizing all sensor extrinsics and time offsets.

| Dimension | MOISST | TLC-Calib |
|------|--------|-----------|
| Scene representation | Single NeRF | Neural Gaussian |
| Multi-sensor | ✓ (arbitrary LiDAR/Camera combinations) | ✓ (multi Camera, single LiDAR) |
| Temporal calibration | ✓ (core feature) | ✓ (optional) |
| Training time | ~25min (KITTI-360) | ~11min |
| Accuracy (R/t) | 4.92°/64.1cm | 0.13°/8.86cm |
| Reference sensor | LiDAR | LiDAR |
| Stability | Moderate (slow NeRF convergence) | High (anchor freezing) |

**MOISST core problem**: Single NeRF must satisfy observation constraints from all sensors simultaneously; when sensor field-of-view differences are large, contradictory gradients arise, causing slow convergence and accuracy degradation.

3DGS-Calib paper data: MOISST rotation error on KITTI-360 is 4.92°±0.5°, far higher than TLC-Calib's 0.13°±0.05°.

#### C. SOAC (Spatio-Temporal Overlap-Aware Calibration, CVPR 2024)

**Core approach**: Improves MOISST by building independent NeRFs per sensor and introducing overlap-aware loss for cross-sensor geometric consistency.

| Dimension | SOAC | TLC-Calib |
|------|------|-----------|
| Scene representation | Multiple NeRFs (one per sensor) | Single Neural Gaussian |
| Overlap awareness | ✓ (core innovation) | Implicit (via anchor sharing) |
| Training time | 1-2h | ~11min |
| Reference sensor | Camera (vs MOISST's LiDAR) | LiDAR |
| Open source | ✓ | ✓ |

### 19.3 NeRF vs 3DGS: Paradigm-Level Differences

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    NeRF vs 3DGS Calibration Paradigm Comparison          │
├────────────────────┬──────────────────────┬──────────────────────────────┤
│ Dimension            │ NeRF-Based           │ 3DGS-Based (TLC-Calib)       │
├────────────────────┼──────────────────────┼──────────────────────────────┤
│ Scene representation │ Continuous implicit  │ Explicit 3D Gaussian points  │
│                      │ MLP: (x,d) → (c,σ)  │ Points: {μ, Σ, α, c}         │
│ Rendering            │ Ray marching         │ Tile-based rasterization     │
│                      │ Integrate along rays │ Per-tile alpha-blending      │
│ Gradient computation │ Automatic diff.      │ CUDA analytic gradients      │
│ Training speed       │ Hours (1-5h)         │ Minutes (~11min)             │
│ Rendering speed      │ Seconds/frame        │ Milliseconds/frame (>100 FPS)│
│ Pose gradient quality│ High (continuous fn) │ High (analytic Jacobian)     │
│ Initialization       │ Requires SfM/LiDAR   │ Direct LiDAR point cloud init│
│ Geometry explicitness│ Low (implicit density)│ High (explicit points+cov)  │
│ Editing/visualization│ Difficult            │ Easy (point cloud operations)│
│ GPU memory           │ High (~16GB)         │ Low (<8GB)                   │
│ Convergence stability│ Moderate (many local minima)│ High (anchor freezing)│
│ Loss landscape smoothness│ Naturally smooth │ Needs Neural Gaussian smoothing│
│ Real-time visualization│ ✗                  │ ✓ (Web Viewer)               │
├────────────────────┴──────────────────────┴──────────────────────────────┤
│ Conclusion: 3DGS leads comprehensively in speed, visualization, editability│
│       NeRF only advantages in natural loss smoothness (TLC-Calib MLP compensates)│
└──────────────────────────────────────────────────────────────────────────┘
```

**NeRF's natural advantage — smooth loss landscape**:

NeRF uses continuous MLP scene representation; continuous input changes produce continuous output changes, naturally yielding smooth loss landscapes. Standard 3DGS discrete Gaussians may produce discontinuous gradients.

**How TLC-Calib compensates**:

Through Neural Gaussian (MLP-generated Gaussian attributes), TLC-Calib brings NeRF's continuity advantage into the 3DGS framework. Paper Figure 5 shows Neural Gaussian produces significantly smoother loss landscapes than standard 3DGS.

### 19.4 Cross-Method Comparison Matrix

```
┌──────────────────────────────────────────────────────────────────────────────┐
│             NeRF + 3DGS Calibration Methods Panorama (2021-2026)             │
├─────────┬───────┬────────┬──────┬────────┬──────────┬──────────┬────────────┤
│ Method  │ Year  │ Repr.  │ Type │ Accuracy│ Train Time│ Multi-cam│ Feature    │
├─────────┼───────┼────────┼──────┼────────┼──────────┼──────────┼────────────┤
│ INF     │ 2023  │ NeRF   │ Joint│ Moderate│ >4h      │ ✗        │ Geometry/  │
│         │       │        │      │        │          │          │ color sep. │
│ MOISST  │ 2023  │ NeRF   │ Joint│ Low    │ ~25min   │ ✓        │ Spatiotemp.│
│         │       │        │      │ 4.92°  │ ~1527s   │          │ multimodal │
│ SOAC    │ 2024  │ NeRF   │ Joint│ Med-high│ 1-2h     │ ✓        │ Overlap    │
│         │       │        │      │        │          │          │ aware multi│
│3DGS-Cal│ 2024  │ 3DGS   │ Joint│ Moderate│ ~9min    │ ✗        │ Hash-Grid  │
│         │       │ +Hash  │      │ 0.45°  │ 216s     │          │ spatiotemp.│
│CalibAny │ 2024  │ Features│ Match│ Med-low│ >5h      │ ✗        │ Zero-train │
│         │       │        │      │ ~1.5°  │          │          │ foundation │
│RobustCal│ 2025  │ 2DGS   │ 2-stg│ Med-high│ >1h      │ ✗        │ Reprojection│
│         │       │        │      │ ~0.3°  │          │          │ +triangulation│
│HiGS-Cal│ 2025  │ 3DGS   │ Decoup│ High   │ -        │ ✗        │ LCPG error │
│         │       │        │      │        │          │          │ hierarchical│
│★TLC-Cal│ 2026  │Neural  │ Joint│ Highest│ ~11min   │ ✓        │ Anchor+Aux │
│         │       │ GS     │+Refn│ 0.13°  │ ~11min   │ Rig opt  │ CUDA grad  │
│         │       │        │      │        │          │          │ AVC+Refine │
└─────────┴───────┴────────┴──────┴────────┴──────────┴──────────┴────────────┘
```

---

## 20. GPU Memory Management Architecture (rasterizer_impl)

> **Source**: `submodules/diff-gaussian-rasterization-w-pose/cuda_rasterizer/rasterizer_impl.h` + `.cu`

### 20.1 Chunk-Based Memory Allocation Pattern

TLC-Calib's CUDA rasterizer uses a **single-allocation + pointer-slicing** memory management pattern, avoiding the overhead of multiple `cudaMalloc` calls:

```
Template function obtain<T>():

  static void obtain(char*& chunk, T*& ptr, size_t count, size_t alignment) {
      // 1. Align to alignment (128 bytes)
      size_t offset = (chunk + alignment - 1) & ~(alignment - 1);
      // 2. Set pointer
      ptr = reinterpret_cast<T*>(offset);
      // 3. Advance chunk pointer
      chunk = reinterpret_cast<char*>(ptr + count);
  }

Usage:
  1. Calculate total requirement: size_t chunk_size = required<GeometryState>(P);
  2. Single allocation:           char* chunkptr = geometryBuffer(chunk_size);
  3. Slice sequentially:          GeometryState::fromChunk(chunkptr, P);
```

### 20.2 Three Major State Buffers

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    GPU Memory Layout (per-frame)                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  GeometryState (per Gaussian, P items)                                   │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ depths      : float    [P]      view-space depth                  │  │
│  │ clamped     : bool     [P*3]    SH color clamping flags           │  │
│  │ internal_radii: int    [P]      screen-space radii                │  │
│  │ means2D     : float2   [P]      2D projected coordinates         │  │
│  │ cov3D       : float    [P*6]    3D covariance upper triangle      │  │
│  │ conic_opacity: float4  [P]      inverse covariance + opacity      │  │
│  │ rgb         : float    [P*3]    RGB from SH evaluation            │  │
│  │ tiles_touched: uint32  [P]      tile count per Gaussian           │  │
│  │ scanning_space: char   [scan]   CUB InclusiveSum workspace        │  │
│  │ point_offsets: uint32  [P]      prefix sum result                 │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│  Total size ≈ P × (4+3+4+8+24+16+12+4+4) + scan ≈ P × 79 bytes        │
│                                                                          │
│  ImageState (per pixel, W×H items)                                       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ accum_alpha : float    [W*H]    final transmittance T_final       │  │
│  │ n_contrib   : uint32   [W*H]    last contributor ID per pixel     │  │
│  │ ranges      : uint2    [tiles]  Gaussian range per tile           │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│  Total size ≈ W×H × 8 + tiles × 8 bytes                                 │
│                                                                          │
│  BinningState (per Gaussian-tile instance, R items)                      │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ point_list_keys_unsorted: uint64 [R]   keys before sort           │  │
│  │ point_list_keys         : uint64 [R]   keys after sort            │  │
│  │ point_list_unsorted     : uint32 [R]   values before sort         │  │
│  │ point_list              : uint32 [R]   values after sort          │  │
│  │ list_sorting_space      : char   [sort] CUB RadixSort workspace   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│  R = Σ tiles_touched[i], Total ≈ R × 24 + sort bytes                    │
│                                                                          │
│  Typical scenario (P=100K, 1920×1080, avg tile coverage=4):              │
│    Geometry: ~7.5 MB                                                     │
│    Image:    ~16 MB                                                      │
│    Binning:  ~9.2 MB                                                     │
│    Total:    ~33 MB (intermediate buffers only, excl. input/output)       │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 20.3 Buffer Reuse Between Forward and Backward

```
Forward:
  geometryBuffer(chunk_size) → allocate → write means2D, depths, cov3D, ...
  binningBuffer(...)         → allocate → sorted point_list
  imageBuffer(...)           → allocate → accum_alpha, n_contrib, ranges

Backward:
  GeometryState::fromChunk(geom_buffer, P)    ← reuse forward geometry buffer
  BinningState::fromChunk(binning_buffer, R)   ← reuse forward sorting results
  ImageState::fromChunk(img_buffer, W*H)       ← reuse forward pixel state

Key: Forward's geomBuffer, binningBuffer, imgBuffer are saved via
     ctx.save_for_backward() and parsed with the same layout in backward,
     avoiding re-allocation and re-computation, saving ~30 MB GPU memory and I/O
```

---

## 21. cameras.py Camera Model and Projection Matrix

> **Source**: `scene/cameras.py` + `utils/graphics_utils.py`

### 21.1 Camera Class State Management

```python
class Camera(nn.Module):
    def __init__(self, cam_id, uid, cam2lidar, lidar_R, lidar_t,
                 GT_cam_R, GT_cam_t, cam_R, cam_t,
                 fx, fy, cx, cy, image, ...):

        # ★ Core identity for calibration optimization
        self.opt_id = cam_id if use_rig else uid
        #   use_rig=True: all frames from same physical camera share opt_id
        #   use_rig=False: each frame gets independent opt_id (for ablation)

        # Pose state: three copies
        self.init_cam2lidar  # Initial LiDAR→Camera extrinsic (from dataset)
        self.init_cam_R/t    # Initial pose (immutable, for pose accumulation)
        self.cam_R/t         # Current pose (updated during optimization)

        # GT pose (evaluation only)
        self.GT_cam_R/t

        # Intrinsics (fixed)
        self.fx, self.fy, self.cx, self.cy
```

### 21.2 Projection Matrix Calculation

TLC-Calib uses **intrinsic-based projection matrices** rather than FoV-based:

```python
# graphics_utils.py
def get_projection_matrix_intrinsics(znear, zfar, cx, cy, fx, fy, W, H):
    # Compute OpenGL-style projection matrix from intrinsics (fx, fy, cx, cy)
    # Supports off-center principal points (cx ≠ W/2, cy ≠ H/2)

    left   = znear / fx * ((2*cx - W)/W - 1.0) * W/2
    right  = znear / fx * ((2*cx - W)/W + 1.0) * W/2
    top    = znear / fy * ((2*cy - H)/H + 1.0) * H/2
    bottom = znear / fy * ((2*cy - H)/H - 1.0) * H/2

    P = | 2n/(r-l)    0      (r+l)/(r-l)     0         |
        |    0     2n/(t-b)   (t+b)/(t-b)     0         |
        |    0        0       zf/(zf-zn)   -zf*zn/(zf-zn)|
        |    0        0          1            0          |
```

**Differences from standard 3DGS**:
- Standard 3DGS uses `get_projection_matrix(fovX, fovY)` — assumes principal point at image center
- TLC-Calib uses `get_projection_matrix_intrinsics(fx, fy, cx, cy)` — supports arbitrary principal point
- Critical for real sensor calibration, as real cameras rarely have principal points exactly at image center

### 21.3 Relationship Between Three Projection Matrices

```python
# Three properties of the Camera class:

@property
def projection_matrix(self):
    # Pure projection matrix P (intrinsics)
    return get_projection_matrix_intrinsics(...).T.cuda()

@property
def world_view_transform(self):
    # View transform matrix V (extrinsics): world → camera
    return get_world_to_view_scaled_torch(self.cam_R, self.cam_t).T

@property
def full_proj_transform(self):
    # Combined matrix V × P: world → NDC
    return (V.unsqueeze(0).bmm(P.unsqueeze(0))).squeeze(0)
```

```
Transformation chain:

  World Space ──[world_view_transform]──→ Camera Space
                     (4×4, contains R, t)

  Camera Space ──[projection_matrix]──→ NDC Space
                     (4×4, contains fx, fy, cx, cy)

  World Space ──[full_proj_transform]──→ NDC Space
                     (4×4, V × P)

CUDA rasterizer receives:
  viewmatrix     = world_view_transform   (for frustum culling, Cov2D)
  projmatrix     = full_proj_transform    (for forward projection to NDC)
  projmatrix_raw = projection_matrix      (for backward pose gradient parsing)
```

### 21.4 MiniCam: Lightweight Camera for Web Viewer

```python
class MiniCam:
    # Dynamically created by ViewerRenderer in viser_utils.py
    # Constructed from the Viewer's interactive camera state
    # Does not include LiDAR pose/GT/intrinsic calibration data
    # Contains only the minimum information needed for rendering
```

---

## 22. train.py vs nvs_eval/train.py Comparison

> **Scope**: TLC-Calib main training script vs standard 3DGS training for NVS evaluation

### 22.1 Design Purpose Differences

```
┌──────────────────────────┬────────────────────────────────────────────────┐
│       TLC-Calib train.py │          nvs_eval/train.py                    │
├──────────────────────────┼────────────────────────────────────────────────┤
│ Goal: Calibrate LiDAR-   │ Goal: Evaluate NVS quality of calibrated      │
│ Camera extrinsics         │ poses                                        │
│ Output: Calibrated        │ Output: PSNR/SSIM/LPIPS metrics              │
│ extrinsic matrix          │                                              │
│ Pose optimization: ✓     │ Pose optimization: ✗ (uses calibrated poses)  │
│ Scene repr: Neural        │ Scene repr: Standard 3DGS (SH)               │
│ Gaussian                  │                                              │
│ Renderer: w-pose custom   │ Renderer: Standard 3DGS renderer             │
│ Rig optimization: ✓      │ Rig optimization: ✗                           │
│ Refine stage: ✓           │ Refine: ✗                                    │
│ Web Viewer: ✓             │ Web Viewer: ✗                                │
└──────────────────────────┴────────────────────────────────────────────────┘
```

### 22.2 Module-by-Module Comparison

| Module | TLC-Calib train.py | nvs_eval/train.py |
|------|---------------------|-------------------|
| **GaussianModel Init** | `GaussianModel(feat_dim, n_offsets, ...)` multi-param Neural Gaussian | `GaussianModel(sh_degree)` standard 3DGS |
| **Color Representation** | MLP-precomputed `colors_precomp` | SH spherical harmonics (up to degree 3) |
| **Position Trainable** | Anchors frozen (lr=0.0) | Fully trainable `fix_xyz=False` |
| **Densification** | Anchor growing/pruning (AVC) | Standard densify_and_prune |
| **Pose Optimization** | `calib_optimizer.step()` + `update_delta_pose()` | None |
| **Loss Function** | L1 + D-SSIM + scale regularization | L1 + D-SSIM |
| **Training Stages** | Main + Refine (two stages) | Single stage |
| **View Sampling** | Random + viewpoint_cycle counter | Random sampling |
| **Pre-filtering** | `prefilter_voxel()` frustum culling | None |
| **Optimizer** | AdamW (scene) + Adam (pose) | Adam |
| **Weight Decay** | First half 1e-2, second half 0 | None |
| **LR Schedule** | Exponential decay (scene) + cosine annealing (pose) | Exponential decay |
| **Render Outputs** | `render`, `depth`, `neural_opacity`, `scaling` | `render`, `viewspace_points`, `radii` |
| **Evaluation** | Convergence curves + pose error | PSNR/SSIM/LPIPS |
| **Output** | Model weights + `cams_to_lidar.txt` | Rendered images + `results.json` |

### 22.3 Independent Pipeline Design for NVS Evaluation

Why not directly use TLC-Calib's rendering results for NVS evaluation?

```
Problem:
  TLC-Calib's Neural Gaussians use MLP colors
  → Color quality is affected by MLP capacity and training strategy
  → Cannot fairly reflect calibration quality

Solution:
  Use calibrated poses → feed to standard 3DGS → retrain scene from scratch
  → NVS quality depends solely on pose accuracy → fair comparison

Pipeline:
  TLC-Calib train.py → cams_to_lidar.txt (calibration result)
       ↓
  nvs_eval/train.py  → load calibrated poses → standard 3DGS training
       ↓
  Render test set → PSNR/SSIM/LPIPS

Key: All comparison methods (3DGS-Calib, RobustCalib, etc.)
     use the same standard 3DGS pipeline for evaluation
     The only variable is the input extrinsics → fair comparison
```

---

## 23. math.h SE(3)/SO(3) CUDA Implementation

> **Source path**: `submodules/diff-gaussian-rasterization-w-pose/cuda_rasterizer/math.h`

### 23.1 Matrix Type Hierarchy

```
mat33 (3×3, column-major float3×3)
  ├── identity()           → identity matrix
  ├── skew_symmetric(v)    → skew-symmetric matrix [v]×
  ├── transpose()          → transpose
  ├── operator+ / * (mat) → matrix addition / multiplication
  └── operator* (float3)  → matrix-vector multiplication

mat34 (3×4, intermediate representation for SE(3))
  └── mat34(mat33, float3) → [R | t]

mat44 (4×4, homogeneous transformation matrix)
  ├── mat44(mat33, float3)  → construct [R t; 0 1] from R, t
  ├── mat44(mat34)          → expand from 3×4
  └── operator* (mat44)    → 4×4 matrix multiplication
```

**Column-major**: `cols[0]` is the first column of the matrix, consistent with OpenGL/CUDA memory layout. In matrix multiplication `A * B`, `A.cols[i]` is the i-th column of A.

### 23.2 SO(3) Rotation Group Implementation

```cpp
struct SO3 {
    mat33 data_;   // internally stores 3×3 rotation matrix

    // Lie algebra → Lie group: Rodrigues formula
    static SO3 Exp(const float3 &theta) {
        mat33 W = hat(theta);        // [θ]× skew-symmetric matrix
        mat33 W2 = W * W;            // [θ]×²
        float angle = norm(theta);   // ||θ||
        mat33 I = mat33::identity();

        // Small-angle approximation (avoids sin/cos numerical instability)
        if (angle < 1e-5)
            return SO3(I + W + 0.5f * W2);

        // Rodrigues formula:
        // R = I + sin(θ)/θ · [θ]× + (1-cos(θ))/θ² · [θ]×²
        return SO3(I + sin(angle)/angle * W
                     + (1-cos(angle))/(angle*angle) * W2);
    }
};
```

**Mathematical correspondence**:

```
Rodrigues formula: R = exp([θ]×) = I + sin(||θ||)/||θ|| · [θ]×
                                + (1-cos(||θ||))/||θ||² · [θ]×²

where [θ]× = | 0   -θz   θy |
             | θz   0   -θx |   (skew-symmetric matrix)
             |-θy   θx   0  |

Small-angle approximation (||θ|| < 1e-5):
  sin(θ)/θ ≈ 1,  (1-cos(θ))/θ² ≈ 0.5
  → R ≈ I + [θ]× + 0.5·[θ]×²
```

### 23.3 SE(3) Rigid Transform Group Implementation

```cpp
struct SE3 {
    SO3 R_data_;     // rotation part
    float3 t_data_;  // translation part

    // SE(3) exponential map: (ρ, θ) → T ∈ SE(3)
    static SE3 Exp(const float3 &rho, const float3 &theta) {
        mat33 W = SO3::hat(theta);
        mat33 W2 = W * W;
        SO3 R = SO3::Exp(theta);
        float angle = norm(theta);
        mat33 I = mat33::identity();

        // Left Jacobian matrix V
        mat33 V;
        if (angle < 1e-5)
            V = I + 0.5f * W + 1.f/6.f * W2;
        else
            V = I + W * ((1-cos(angle))/(angle*angle))
                  + W2 * ((angle-sin(angle))/(angle*angle*angle));

        // Translation = V · ρ (not ρ directly!)
        float3 t = V * rho;
        return SE3(t, R);
    }

    // SE(3) composition (left multiplication)
    SE3 operator*(const SE3 &T) const {
        return SE3(t + R * T.t, R * T.R);
    }

    // Inverse transform
    SE3 inverse() const {
        SO3 R_inv = R.inverse();      // R^T
        float3 t = R_inv * t_data_;   // -R^T · t
        return SE3(-t, R_inv);
    }
};
```

**Key mathematics**: In the SE(3) exponential map, the translation part does **not** use ρ directly, but is transformed through the **left Jacobian matrix V**:

```
SE(3) Exp:  T = exp([τ]∧) = | R    V·ρ |
                             | 0     1  |

Left Jacobian matrix:
  V = I + (1-cos(θ))/θ² · [θ]× + (θ-sin(θ))/θ³ · [θ]×²

Small-angle approximation:
  V ≈ I + 0.5·[θ]× + 1/6·[θ]×²

τ = [ρ; θ] ∈ se(3) is the 6-dimensional Lie algebra vector:
  ρ = [ρ1, ρ2, ρ3]  → translation direction (transformed by V)
  θ = [θ1, θ2, θ3]  → rotation axis-angle
```

### 23.4 CUDA vs Python Dual Implementation Comparison

```
┌─────────────────────┬────────────────────────┬─────────────────────────┐
│       Feature       │  CUDA (math.h)         │  Python (pose_utils.py) │
├─────────────────────┼────────────────────────┼─────────────────────────┤
│ SO3 Exp             │ SO3::Exp()             │ _SO3_exp()              │
│ SE3 Exp             │ SE3::Exp()             │ SE3_exp()               │
│ Left Jacobian V     │ inlined in SE3::Exp()  │ _SO3_left_jacobian()    │
│ Skew-symmetric      │ mat33::skew_symmetric()│ _skew_symmetric()       │
│ Small-angle threshold│ 1e-5                   │ 1e-5                    │
│ Use case            │ backward gradient propagation│ forward pose update │
│ Autodiff            │ ✗ (manual analytic gradients)│ ✓ (PyTorch autograd) │
│ Precision           │ float (32-bit)         │ float (32-bit)          │
│ Storage format      │ column-major mat33/mat44│ torch.Tensor            │
│ Matrix multiplication│ hand-expanded          │ @ operator              │
└─────────────────────┴────────────────────────┴─────────────────────────┘
```

**Why two implementations?**
- The CUDA version is used for **backward propagation**: In `backward.cu`, the SE(3) transform matrix must be constructed from τ (Lie algebra) to compute pose gradients `dL/dτ`. The Exp operation here is part of the forward computation graph; gradients are computed via hand-written analytic formulas
- The Python version is used for **forward updates**: In `poses.py → update_delta_pose()`, the optimizer produces Lie algebra increments δτ, which must be converted to transform matrix ΔT via `SE3_exp(δτ)`, then left-multiplied onto the current pose

### 23.5 Key Role of SE(3) in Gradient Propagation

```
Pose gradient propagation chain:

  Loss ──[dL/dC]──→ rendered output C
       ──[dC/d(μ2D)]──→ 2D projected coordinates μ2D
       ──[d(μ2D)/dτ]──→ pose Lie algebra τ ∈ se(3)

In backward.cu:
  1. Construct current pose: SE3 T_cur(tau_data)  // build from 6-dim τ
  2. Transform 3D points:   p_cam = T_cur * p_world
  3. Compute Jacobian:      d(p_cam)/dτ = [I | -[p_cam]×]  (SE(3) adjoint representation)
  4. Accumulate gradients:  dL/dτ += dL/d(p_cam) · d(p_cam)/dτ

Analytic form of d(p_cam)/dτ:
  For τ = [ρ; θ]:
    d(p_cam)/dρ = R       (rotation matrix, translation direction gradient)
    d(p_cam)/dθ = -R·[p]×  (rotation differential w.r.t. 3D point)
```

---

## 24. dataset_readers.py Dataset Format Parsing Differences

> **Source path**: `scene/dataset_readers.py` + `utils/pose_utils.py`

### 24.1 Unified Data Directory Structure

All datasets follow a unified directory structure:

```
dataset_root/
├── params/
│   ├── cams_to_lidar_gt.txt    # GT extrinsics: cam_id + 4×4 matrix (per row)
│   ├── intrinsics.txt          # intrinsics: 3×3 matrix per camera
│   ├── lidars.txt              # LiDAR pose sequence: 4×4 matrix per row
│   └── timestamps.txt          # LiDAR timestamps (optional)
├── images/
│   ├── image_00/               # image sequence for camera 0
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   └── image_01/               # image sequence for camera 1 (multi-camera)
└── lidar/
    └── map.ply                 # aggregated LiDAR point cloud map
```

### 24.2 From-LiDAR vs From-Blueprint Initialization

The **core difference** across datasets lies in the source of initial extrinsics:

```python
# In read_custom_cameras():
if from_lidar:
    # ★ From-LiDAR: use coarse initial extrinsics based on dataset type
    init_cam2lidar = get_c2l(cam_id, dataset_type)
    # get_c2l() returns coarse C2L based on dataset priors
else:
    # ★ From-Blueprint: use GT extrinsics (+ optional noise)
    init_cam2lidar = cam2lidar  # from cams_to_lidar_gt.txt
    if noises is not None:
        noise_lc2w = lc2w @ noises[cam_id]  # inject noise
        w2lc = np.linalg.inv(noise_lc2w)
```

### 24.3 get_c2l(): Dataset Prior Extrinsics

```python
# Dataset prior configuration in pose_utils.py:

BASE_CAMERA_TO_LIDAR = np.array([
    [0.0,  0.0, 1.0, 0.0],   # Camera Z → LiDAR X (forward)
    [-1.0, 0.0, 0.0, 0.0],   # Camera X → LiDAR -Y (left)
    [0.0, -1.0, 0.0, 0.0],   # Camera Y → LiDAR -Z (up)
    [0.0,  0.0, 0.0, 1.0],
])

DATASET_CAMERA_YAW_DEGREES = {
    "kitti-360": {2: 90.0, 3: -90.0},     # Cam2=right 90°, Cam3=left 90°
    "kitti":     {},                        # forward camera only, no yaw
    "waymo":     {1: 45.0, 2: -45.0,       # front-left 45°, front-right -45°
                  3: 90.0, 4: -90.0},      # side-left 90°, side-right -90°
    "fast-livo2": {},                       # single camera, no yaw
}
```

**How it works**: `get_c2l(cam_id, dataset_type)` first takes `BASE_CAMERA_TO_LIDAR` (assuming forward-facing camera), then applies yaw rotation based on dataset and camera ID. This is a **coarse prior**; actual extrinsics require TLC-Calib optimization.

### 24.4 Specific Differences Across Three Datasets

```
┌────────────────┬─────────────────┬──────────────────┬───────────────────┐
│                │   KITTI-360     │     Waymo        │   FAST-LIVO2      │
├────────────────┼─────────────────┼──────────────────┼───────────────────┤
│ Camera count   │ 2 (Cam2, Cam3) │ 5 (Cam0-4)      │ 1                 │
│ Camera facing  │ left/right 90° │ front+front-left/right+side left/right│ forward │
│ LiDAR rate     │ 10 Hz          │ 10 Hz            │ 10 Hz             │
│ Image format   │ PNG            │ PNG              │ PNG               │
│ Timestamps     │ ✓ (file provided)│ ✓ (file provided)│ ✓ (file provided)│
│ Point cloud map│ aggregated PLY │ aggregated PLY   │ aggregated PLY    │
│ Intrinsics     │ pinhole model  │ pinhole model    │ pinhole model     │
│ From-LiDAR prior│ yaw angles    │ yaw angles       │ no yaw (direct forward)│
│ GT extrinsic source│ official calibration│ official calibration│ official calibration│
│ Test split     │ every 2 frames │ every 2 frames │ every 2 frames    │
│ Typical frames │ 50-300         │ 100-200          │ 50-200            │
└────────────────┴─────────────────┴──────────────────┴───────────────────┘
```

### 24.5 Time Offset Handling

```python
# Time offset injection (simulates LiDAR-Camera time misalignment):
if time_offset != 0:
    time_offset_list = [time_offset * np.random.choice([-1, 1])
                        for _ in range(cam_number)]

    for cam_id in range(cam_number):
        if lidar_timestamps is not None:
            # Use real timestamps: SLERP interpolation + linear extrapolation
            lidar_extrinsics_offset[cam_id] = apply_time_offset(
                lidar_extrinsics[cam_id],
                time_offset_list[cam_id] * 0.001,  # ms → s
                lidar_timestamps=lidar_timestamps)
        else:
            # Use uniform spacing: assume 10Hz
            lidar_extrinsics_offset[cam_id] = apply_time_offset(
                lidar_extrinsics[cam_id],
                time_offset_list[cam_id] * 0.001,
                frame_rate=frame_rate)
```

**Interpolation/extrapolation strategy (pose_utils.py)**:
- **Interpolation**: target time falls between two adjacent frames → SLERP (spherical linear interpolation for rotation) + linear interpolation for translation
- **Extrapolation**: target time exceeds range → constant-velocity extrapolation using boundary frame angular and linear velocity

### 24.6 Adaptive Voxel Control (AVC) Voxel Size Computation

```python
if adaptive_voxel:
    # Adaptively compute voxel size based on trajectory length
    traj_length = get_traj_length(lidar_poses)
    target_voxels = int(traj_length * avc_beta)  # beta=5000
    voxel_size = compute_voxel_size(pc, target_voxels=target_voxels)

# compute_voxel_size(): binary search
# Search in [0.1, 0.5] for voxel_size such that voxel_count ≈ target_voxels
# Tolerance 5%, max 30 iterations
```

**Design intent**: Short-trajectory scenes (e.g., straight driving) need denser points (small voxels); long-trajectory scenes (large turns) have more dispersed point clouds and can use large voxels. Dynamically adjust target voxel count via `traj_length × beta`.

---

## 25. gaussian_model.py Neural Gaussian Complete Initialization and Optimization

> **Source path**: `scene/gaussian_model.py`

### 25.1 Neural Gaussian Parameter Composition

```
Trainable parameters of GaussianModel:

  Anchor parameters (one per Anchor Gaussian):
  ┌─────────────────────────────────────────────────────────────┐
  │ _anchor      : [N, 3]        Anchor 3D position           │
  │ _anchor_feat : [N, feat_dim] Anchor features (default feat_dim=32)│
  │ _scaling     : [N, 6]        6D scale (3 for anchor + 3    │
  │                               for offset range)             │
  │ _rotation    : [N, 4]        quaternion rotation            │
  │ _opacity     : [N, 1]        Anchor opacity                 │
  │ _offset      : [N, k, 3]    offset vectors (k=n_offsets, default 5)│
  └─────────────────────────────────────────────────────────────┘

  MLP network parameters:
  ┌─────────────────────────────────────────────────────────────┐
  │ mlp_opacity : Linear(feat+3, feat) → ReLU → Linear(feat, k)│
  │              → Tanh      output: opacity of k Auxiliary Gaussians│
  │ mlp_cov     : Linear(feat+3, feat) → ReLU → Linear(feat,7k)│
  │              output: k Auxiliary (scale[3]+rotation[4])     │
  │ mlp_color   : Linear(feat+3+app, feat) → ReLU              │
  │              → Linear(feat, 3k) → Sigmoid                   │
  │              output: RGB color of k Auxiliary Gaussians     │
  └─────────────────────────────────────────────────────────────┘

  Optional modules:
  ┌─────────────────────────────────────────────────────────────┐
  │ embedding_appearance : Embedding(num_cams, app_dim)         │
  │                        per-camera appearance embedding (default app_dim=32)│
  │ mlp_feature_bank     : Linear(4, feat) → ReLU → Linear(3)  │
  │                        → Softmax  (optional, disabled by default)│
  └─────────────────────────────────────────────────────────────┘
```

### 25.2 Point Cloud Initialization (create_from_pcd)

```python
def create_from_pcd(self, pcd, voxel_size, spatial_lr_scale):
    # 1. Set voxel size (if <= 0, use KNN median distance)
    if self.voxel_size <= 0:
        init_dist = distCUDA2(init_points)  # CUDA KNN
        median_dist = kthvalue(init_dist, int(N*0.5))
        self.voxel_size = median_dist

    # 2. Anchor position = point cloud coordinates (LiDAR frame)
    self._anchor = Parameter(points)

    # 3. Offsets initialized to zero (learned during training)
    self._offset = Parameter(zeros([N, n_offsets, 3]))

    # 4. Features initialized to zero
    self._anchor_feat = Parameter(zeros([N, feat_dim]))

    # 5. Scale = log(sqrt(KNN distance)), 6-dim (3+3)
    dist2 = distCUDA2(points)  # squared distance to nearest neighbor per point
    scales = log(sqrt(dist2)).repeat(1, 6)  # [N, 6]

    # 6. Rotation = identity quaternion [1,0,0,0]
    rots = zeros([N, 4])
    rots[:, 0] = 1

    # 7. Opacity = inverse_sigmoid(0.1) ≈ -2.197
    opacities = inverse_sigmoid(0.1)
```

### 25.3 Optimizer Configuration and LR Scheduling

```
Scene optimizer (AdamW):
┌──────────────────┬────────────────────┬──────────────┐
│ Parameter group  │ Initial LR         │ Schedule     │
├──────────────────┼────────────────────┼──────────────┤
│ anchor           │ position_lr × SLS  │ exponential decay│
│ offset           │ offset_lr × SLS    │ exponential decay│
│ anchor_feat      │ feature_lr         │ fixed        │
│ opacity          │ opacity_lr         │ fixed        │
│ scaling          │ scaling_lr         │ fixed        │
│ rotation         │ rotation_lr        │ fixed        │
│ mlp_opacity      │ mlp_opacity_lr     │ exponential decay│
│ mlp_cov          │ mlp_cov_lr         │ exponential decay│
│ mlp_color        │ mlp_color_lr       │ exponential decay│
│ embedding_app    │ appearance_lr      │ exponential decay│
└──────────────────┴────────────────────┴──────────────┘
SLS = spatial_lr_scale (scene spatial scaling factor)

Pose optimizer (Adam, one per optimization ID):
┌──────────────────┬────────────────────┬──────────────┐
│ Parameter group  │ Initial LR         │ Schedule     │
├──────────────────┼────────────────────┼──────────────┤
│ cam_rot_delta    │ calib_rot_lr_init  │ cosine annealing│
│ cam_trans_delta  │ calib_trans_lr_init│ cosine annealing│
└──────────────────┴────────────────────┴──────────────┘
```

### 25.4 Pose State Management

```python
def init_cam(self, cam_num):
    # Rotation increment per optimization ID (se(3) space)
    self.cam_rot_deltas = ParameterList([
        Parameter(zeros(3)) for _ in range(cam_num)
    ])
    # Translation increment per optimization ID
    self.cam_trans_deltas = ParameterList([
        Parameter(zeros(3)) for _ in range(cam_num)
    ])
    # Accumulated transform matrices (numpy)
    self.accum_P = [eye(4) for _ in range(cam_num)]
    # Predicted Rig (C2L)
    self.rigs = [eye(4) for _ in range(cam_num)]

# After each optimization step:
def update_delta_pose(self, cam_idx, delta_P):
    # Left-multiply accumulation: P_new = ΔP · P_old
    self.accum_P[cam_idx] = delta_P @ self.accum_P[cam_idx]
```

### 25.5 Anchor Growing and Pruning (AVC)

```
Anchor Growing pipeline (adjust_anchor + anchor_growing):

  1. Compute offset gradients: grads = offset_gradient_accum / offset_denom
  2. Multi-level growth (update_depth=3 levels):
     Level 0: threshold = grad_threshold
     Level 1: threshold × (hierachy_factor/2)
     Level 2: threshold × (hierachy_factor/2)²

  3. Per level:
     a. Candidate selection: grad >= threshold AND sufficient observation count
     b. Random sampling: probability = 0.5^(level+1) (fewer at deeper levels)
     c. Coordinate quantization: round(xyz / cur_size) → grid coordinates
     d. Deduplication: exclude existing Anchor grid positions
     e. New Anchor:
        - position = quantized grid × cur_size
        - scale = log(cur_size)
        - rotation = [1,0,0,0]
        - opacity = inverse_sigmoid(0.1)
        - features = source Anchor features (scatter_max aggregation)

Anchor Pruning:
  Condition: opacity_accum < min_opacity × anchor_demon
         AND anchor_demon > check_interval × success_threshold
  i.e.: Anchors frequently observed but always low-opacity are removed
```

---

## 26. loss_utils.py Loss Functions and Gradient Contribution Deep Analysis

> **Source path**: `utils/loss_utils.py` + `train.py`

### 26.1 Total Loss Function Composition

```python
# In train.py:
loss = (1 - λ_dssim) × L_photo + λ_dssim × L_ssim + λ_scale × L_scale

# Default parameters:
#   λ_dssim = 0.2
#   λ_scale = 1.0
#   scale_regularizer = 10.0
```

### 26.2 Photometric Loss L_photo (L1)

```python
def l1_loss(network_output, gt):
    return torch.abs(network_output - gt).mean()

# Gradient analysis:
# dL/dC_pred = sign(C_pred - C_gt) / (H × W × 3)
#
# Characteristics:
# - Robust to outliers (gradient constant at ±1/N)
# - Does not amplify large errors (vs L2 linear amplification)
# - Weight: (1 - 0.2) = 0.8
```

### 26.3 Structural Similarity Loss L_ssim (D-SSIM)

```python
def ssim(img1, img2, window_size=11, size_average=True):
    # 1. Create Gaussian window (σ=1.5)
    window = gaussian_window(window_size=11, sigma=1.5)

    # 2. Compute local statistics (via convolution)
    mu1 = conv2d(img1, window)      # local mean
    mu2 = conv2d(img2, window)
    sigma1_sq = conv2d(img1², window) - mu1²  # local variance
    sigma2_sq = conv2d(img2², window) - mu2²
    sigma12   = conv2d(img1×img2, window) - mu1×mu2  # covariance

    # 3. SSIM formula
    C1, C2 = 0.01², 0.03²
    ssim = (2·mu1·mu2+C1)(2·sigma12+C2) /
           (mu1²+mu2²+C1)(sigma1²+sigma2²+C2)

    # 4. D-SSIM = (1 - SSIM) / 2
    return ssim.mean()

# Gradient analysis:
# dL_ssim/dC_pred involves:
#   - dSSIM/dmu1 (effect of local mean change on SSIM)
#   - dSSIM/dsigma12 (effect of local covariance on SSIM)
#   - Backprop through conv2d, producing spatially smoothed gradient field
#
# Physical meaning:
#   - L1 focuses on pixel-level error
#   - SSIM focuses on structure-level error (brightness + contrast + structure)
#   - Combined: balances pixel accuracy and perceptual quality
#
# Weight: 0.2
```

### 26.4 Scale Regularization L_scale

```python
# In train.py:
scaling = render_pkg["scaling"]  # [P_visible, 6]
if scaling.shape[0] > 0:
    scaling_reg = (
        clamp(scaling.max()/scaling.min() - scale_regularizer, min=0)
        .view(-1, 1).sum()
    ) / scaling.shape[0]

# Interpretation:
# scaling.max() / scaling.min() = max scale / min scale (condition number)
# If condition number > 10 (scale_regularizer), penalize excess
# Normalized: divide by visible Gaussian count
```

```
Gradient contribution of scale regularization:

  When max/min > 10:
    dL_scale/d(scaling_max) = 1/min / N_visible
    dL_scale/d(scaling_min) = -max/min² / N_visible

  Physical meaning:
    Prevents Gaussians from becoming overly anisotropic (extremely long in one direction, short in another)
    → suppresses "needle-like" degeneration
    → maintains reasonable ellipsoid shape
    → indirectly improves pose gradient quality (anisotropic Gaussians yield unstable pose gradients)

  Meaning of threshold 10:
    Allows max/min scale ratio of 10:1
    Linear penalty beyond that
    Weight λ_scale=1.0 (same order as photometric loss)
```

### 26.5 Gradient Flow Comparison of Three Loss Terms

```
┌─────────────┬──────────────────────┬───────────────────────────────────┐
│ Loss term   │ Gradient flow        │ Effect on pose optimization       │
├─────────────┼──────────────────────┼───────────────────────────────────┤
│ L1 (0.8)    │ rendered pixels → 2D mean│ Direct drive: pixel error → pose adjustment│
│             │ → Cov2D → pose τ     │ Large gradients but direction may be noisy│
├─────────────┼──────────────────────┼───────────────────────────────────┤
│ D-SSIM (0.2)│ SSIM map → conv2d backprop│ Indirect drive: structure error → smoothed spatial gradients│
│             │ → rendered pixels → τ│ Provides smooth gradient signal, avoids local minima│
├─────────────┼──────────────────────┼───────────────────────────────────┤
│ L_scale(1.0)│ scaling → no direct pose│ Indirect: constrain Gaussian shape → improve│
│             │   gradient           │ Cov2D gradient quality in subsequent iterations│
└─────────────┴──────────────────────┴───────────────────────────────────┘
```

---

## 27. Real-Vehicle Deployment and Cross-Vehicle Generalization Audit

> **Audit scope**: Whether TLC-Calib's current training framework can be deployed on real vehicles and generalize real-time calibration across vehicle types

### 27.1 Current Framework Design Positioning

```
TLC-Calib design goals:
  ✓ Offline/quasi-offline calibration (not real-time inference)
  ✓ Targetless
  ✓ Arbitrary driving scenes (no scene-type restriction)
  ✗ Not a real-time online calibration system

Key metrics:
  Training time: ~10 minutes / scene (RTX 4090)
  GPU memory: < 8 GB
  Input requirement: 50-300 frames LiDAR + Camera data
  Output: 4×4 extrinsic matrix (camera-to-lidar)
```

### 27.2 Real-Vehicle Deployment Feasibility Analysis

```
┌───────────────────────┬──────────┬──────────────────────────────────────┐
│ Evaluation dimension  │ Rating   │ Detailed analysis                    │
├───────────────────────┼──────────┼──────────────────────────────────────┤
│ Hardware requirements │ ⚠ Medium │ Requires GPU (≥RTX 3060), not suitable│
│                       │          │ for embedded ECU, but vehicle GPU    │
│                       │          │ platforms supported (e.g. NVIDIA Orin/Xavier)│
├───────────────────────┼──────────┼──────────────────────────────────────┤
│ Real-time capability  │ ✗ Not met│ 10-minute training does not meet real-time needs; only suitable for:│
│                       │          │ - offline calibration at vehicle startup│
│                       │          │ - periodic maintenance calibration   │
│                       │          │ - post-processing calibration verification│
├───────────────────────┼──────────┼──────────────────────────────────────┤
│ Automation level      │ ✓ High   │ Fully targetless, no manual intervention│
│                       │          │ full_eval.py supports batch automation │
├───────────────────────┼──────────┼──────────────────────────────────────┤
│ Data collection ease  │ ✓ Low    │ Normal driving data suffices, no special scenes│
│                       │          │ 50+ frames (~5 seconds) sufficient for calibration│
├───────────────────────┼──────────┼──────────────────────────────────────┤
│ Point cloud map dependency│ ⚠ Preprocessing needed│ Requires SLAM/Odometry aggregated map first│
│                       │          │ Adds upfront pipeline complexity     │
├───────────────────────┼──────────┼──────────────────────────────────────┤
│ Environment dependency│ ⚠ Medium │ PyTorch + CUDA + custom CUDA kernels │
│                       │          │ Deployment needs build environment, but can be Dockerized│
└───────────────────────┴──────────┴──────────────────────────────────────┘
```

### 27.3 Cross-Vehicle Generalization Capability Assessment

```
┌───────────────────────┬──────────┬──────────────────────────────────────┐
│ Generalization dimension│ Rating   │ Detailed analysis                    │
├───────────────────────┼──────────┼──────────────────────────────────────┤
│ Different camera counts │ ✓ Supported│ cam_num auto-inferred from data, Rig optimization│
│ (1-5+)               │          │ supports arbitrary camera count; Waymo 5 cameras verified│
├───────────────────────┼──────────┼──────────────────────────────────────┤
│ Different camera facing │ ✓ Supported│ DATASET_CAMERA_YAW_DEGREES configurable│
│ (front/side/rear)    │          │ get_c2l() provides coarse prior      │
│                       │          │ actual extrinsics learned by optimizer│
├───────────────────────┼──────────┼──────────────────────────────────────┤
│ Different LiDAR models│ ✓ Supported│ Only needs PLY point cloud map, no specific sensor dependency│
│                       │          │ KITTI/Waymo/FAST-LIVO2 verified      │
├───────────────────────┼──────────┼──────────────────────────────────────┤
│ Different resolutions   │ ✓ Supported│ Intrinsics (fx,fy,cx,cy) read from config│
│                       │          │ resolution parameterized, not hardcoded│
├───────────────────────┼──────────┼──────────────────────────────────────┤
│ New dataset integration│ ⚠ Adaptation needed│ Requires standard params/ directory format│
│                       │          │ Register new dataset priors in       │
│                       │          │ DATASET_CAMERA_YAW_DEGREES           │
│                       │          │ or use from_lidar=False + noise      │
├───────────────────────┼──────────┼──────────────────────────────────────┤
│ Fisheye/wide-angle lenses│ ✗ Not supported│ Pinhole model (PINHOLE) only        │
│                       │          │ Requires camera model and rasterizer extension│
├───────────────────────┼──────────┼──────────────────────────────────────┤
│ Multi-LiDAR fusion    │ ⚠ Not verified│ Code assumes single LiDAR reference frame│
│                       │          │ Multi-LiDAR requires Rig optimization extension│
├───────────────────────┼──────────┼──────────────────────────────────────┤
│ Initial noise tolerance│ ✓ Strong │ Paper verified: translation ±1m, rotation ±10° converges│
│                       │          │ Most vehicle mounting deviations far smaller│
└───────────────────────┴──────────┴──────────────────────────────────────┘
```

### 27.4 Core Bottlenecks and Improvement Recommendations

```
Bottleneck 1: Non-real-time → cannot calibrate online
  Current: 10 minutes / scene, requires offline processing
  Improvement directions:
    - Reduce iteration count (current 30000, explore early stopping)
    - Lightweight Gaussian representation (reduce Anchor count)
    - TensorRT acceleration of CUDA kernels
    - Use pretrained initialization (reduce convergence time)

Bottleneck 2: Requires LiDAR point cloud map
  Current: Must pre-build aggregated map (SLAM/Odometry)
  Improvement directions:
    - Support incremental point cloud input
    - Integrate lightweight SLAM frontend
    - Use depth prediction to replace LiDAR initialization

Bottleneck 3: Pinhole camera only
  Current: No support for fisheye/wide-angle/panoramic cameras
  Improvement directions:
    - Extend CUDA kernels for distortion models
    - Pre-undistortion (simple but loses information)
    - Adopt universal camera models (e.g. UCM/DS)

Bottleneck 4: Manual adaptation for new vehicle types
  Current: Requires standard directory structure + prior registration
  Improvement directions:
    - Auto-detect sensor configuration
    - Unified data interface (ROS bag / Protobuf)
    - Provide GUI configuration tool

Bottleneck 5: Single-shot calibration (no continuous monitoring)
  Current: Each calibration independent, cannot track drift
  Improvement directions:
    - Integrate temporal extrinsic monitoring
    - Extrinsic drift detection + auto-trigger re-calibration
    - Online filtering (EKF/UKF) hybrid with TLC-Calib
```

### 27.5 Recommended Real-Vehicle Deployment Schemes

```
Scheme A: Offline calibration station (recommended)
  ┌────────────────────────────────────────────────┐
  │  Vehicle enters → collect 30s data → upload to cloud/local GPU│
  │  → TLC-Calib auto calibration (~10min)         │
  │  → extrinsics written to vehicle config → done │
  └────────────────────────────────────────────────┘
  Use cases: factory delivery, periodic maintenance, post-collision re-calibration

Scheme B: Startup self-calibration
  ┌────────────────────────────────────────────────┐
  │  Vehicle startup → first 30s normal driving data → background optimization│
  │  → use old extrinsics until calibration completes│
  │  → hot-update extrinsics when done             │
  └────────────────────────────────────────────────┘
  Requirements: onboard GPU (Orin/Xavier), wait on first startup

Scheme C: Edge computing calibration service
  ┌────────────────────────────────────────────────┐
  │  Vehicle periodically uploads driving data → edge/cloud GPU calibration│
  │  → OTA push new extrinsics                     │
  └────────────────────────────────────────────────┘
  Use cases: fleet management, large-scale deployment

Comparison:
  │ Scheme │ Latency │ Cost │ Onboard GPU │ Use case        │
  ├──────┼──────┼──────┼────────────┼────────────┤
  │  A   │ minutes│ low  │ ✗          │ offline maintenance│
  │  B   │ minutes│ high │ ✓          │ high-end autonomous driving│
  │  C   │ hours  │ medium│ ✗          │ fleet operations│
```

### 27.6 Summary Assessment

```
┌─────────────────────────────────────────────────────────────────────┐
│              TLC-Calib Real-Vehicle Deployment Overall Score        │
├──────────────────────┬──────────────────────────────────────────────┤
│ Calibration accuracy │ ★★★★★  SOTA level, 0.13°/8.86cm              │
│ Robustness (initial noise)│ ★★★★☆  ±1m/±10° converges, covers most scenarios│
│ Sensor compatibility │ ★★★★☆  pinhole camera + arbitrary LiDAR, no fisheye support│
│ Multi-camera support │ ★★★★★  Rig optimization, 1-5+ cameras verified│
│ Automation level     │ ★★★★☆  targetless fully automatic, but needs pre-built point cloud map│
│ Deployment convenience│ ★★★☆☆  requires GPU + CUDA build environment │
│ Real-time capability │ ★★☆☆☆  offline/quasi-offline, not online real-time calibration│
│ Cross-vehicle generalization│ ★★★★☆  minor adaptation needed (prior config), framework generic│
│ Production readiness │ ★★★☆☆  research-grade code, needs engineering (error handling/logging etc.)│
├──────────────────────┼──────────────────────────────────────────────┤
│ Overall assessment   │ Suitable as offline/quasi-offline calibration tool for real vehicles│
│                      │ Not suitable as online real-time calibration system│
│                      │ Cross-vehicle generalization requires standardized data interface│
└──────────────────────┴──────────────────────────────────────────────┘
```

---

## 28. Experimental Results and Analysis

### 28.1 KITTI-360 Calibration Accuracy

| Scene | Success Rate | Rotation Error (°) | Translation Error (cm) |
|------|--------|------------|-------------|
| Straight | 100% | 0.11±0.04 | 12.7±1.43 |
| Small zigzag | 100% | 0.15±0.02 | 10.1±0.73 |
| Large zigzag | 100% | 0.09±0.03 | 6.17±2.27 |
| Small rotation | 100% | 0.21±0.03 | 6.21±1.67 |
| Large rotation | 100% | 0.09±0.03 | 9.18±1.04 |
| **Average** | **100%** | **0.13±0.05** | **8.86±2.90** |

vs. strongest baseline (INF): SR 74.5%, R 0.41°, t 32.6cm

### 28.2 NVS Performance

| Dataset | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| KITTI-360 average | **26.39** | **0.85** | **0.09** |
| WAYMO average | **27.04** | **0.86** | **0.11** |
| FAST-LIVO2 average | **21.90** | 0.656 | 0.205 |

### 28.3 Training Efficiency

| Method | Training Time | GPU Memory |
|------|---------|---------|
| CalibAnything | >5h | - |
| INF | >4h | - |
| RobustCalib | >1h | - |
| 3DGS-Calib* | ~0.15h | RTX 5090 |
| **TLC-Calib** | **~0.18h** | **<8GB (RTX 4090)** |

### 28.4 Key Ablation Findings

| Ablation | Conclusion |
|---------|------|
| Rig vs Per-image | Remove rig: 0.13°→1.94°, 8.86cm→64.7cm (collapse) |
| Auxiliary Gaussians | Remove auxiliary: NVS quality drops significantly, calibration accuracy degrades |
| AVC | Fixed voxel: inconsistent performance across scenes |
| L_scale | Remove regularization: training unstable |
| Full image vs crop | Full image better: auxiliary Gaussians utilize LiDAR blind zones |
| Extrinsic generalization | Scene A optimization → Scene B transfer succeeds |

---

## 29. Complete Code Architecture and Data Flow

```
TLC-Calib/
├── train.py                    # Main training loop: joint scene+pose optimization
├── render.py                   # Rendering: visualize trained calibration model
├── metrics_pose.py             # Pose evaluation: rotation/translation error (APE)
├── metrics_nvs.py              # NVS evaluation: calls nvs_eval submodule
├── full_eval.py                # Full evaluation: batch calibration+eval+aggregation
│
├── scene/
│   ├── __init__.py             # Scene class: data loading, model management
│   ├── gaussian_model.py       # GaussianModel: anchor/auxiliary, MLP,
│   │                           #   optimizers, pose state, rig management
│   ├── poses.py                # Pose updates: SE3 incremental accumulation, rig updates
│   ├── embedding.py            # Appearance embedding (per-image, disabled by default)
│   ├── dataset_readers.py      # Data reading: KITTI-360/Waymo/FAST-LIVO2,
│   │                           #   AVC, time offset, noise injection
│   ├── cameras.py              # Camera class: intrinsics/extrinsics, projection matrices
│   └── colmap_loader.py        # COLMAP format loader
│
├── gaussian_renderer/
│   └── __init__.py             # Rendering: neural gaussian generation +
│                                #   pose-differentiable rasterization + voxel prefiltering
│
├── submodules/
│   ├── diff-gaussian-rasterization-w-pose/  # Custom CUDA rasterizer
│   │   ├── cuda_rasterizer/
│   │   │   ├── forward.cu      # Forward: 3D→2D projection, alpha compositing
│   │   │   ├── backward.cu     # Backward: four gradient paths to pose
│   │   │   ├── math.h          # SE3/SO3 Lie group operations (new)
│   │   │   └── auxiliary.h     # Auxiliary functions
│   │   ├── rasterize_points.cu # PyTorch ↔ CUDA interface
│   │   └── diff_gaussian_rasterization_w_pose/
│   │       └── __init__.py     # Python binding (extracts grad_theta/grad_rho)
│   └── simple-knn/             # KNN distance computation
│
├── utils/
│   ├── pose_utils.py           # SE3/SO3 math, APE metrics,
│   │                           #   time offset interpolation, coordinate transforms
│   ├── loss_utils.py           # L1/L2/SSIM losses
│   ├── noise_utils.py          # Initialization noise generation (for ablations)
│   ├── camera_utils.py         # Camera utility functions
│   ├── graphics_utils.py       # Projection matrices, FoV conversion
│   ├── viser_utils.py          # Web viewer integration (viser + nerfview)
│   └── general_utils.py        # General utilities (lr schedulers: exp/cosine/warmup)
│
├── arguments/__init__.py       # All hyperparameter definitions (consistent with paper)
│
├── nvs_eval/                   # Independent standard 3DGS NVS evaluation module
│   ├── train.py                # Standard 3DGS training + rendering + evaluation
│   ├── scene/                  # Scene management (supports prior_pose)
│   ├── gaussian_renderer/      # Standard renderer (no pose gradients)
│   ├── utils/                  # Utility functions
│   └── submodules/             # Standard 3DGS submodules
│       ├── diff-gaussian-rasterization/
│       └── simple-knn/
│
└── lpipsPyTorch/               # LPIPS perceptual loss computation
```

---

## 30. Summary of Key Design Insights

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    TLC-Calib Ten Key Design Insights                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ❶ Delicate division between frozen vs learnable                          │
│     Anchor positions frozen → global scale does not drift                 │
│     MLP parameters learnable → local geometry+appearance adaptation     │
│     Solves dual dilemma: "pose optimization causes scene drift" and       │
│     "scene overfitting traps pose in local minima"                        │
│                                                                          │
│  ❷ Full-image supervision > region cropping                               │
│     Auxiliary Gaussians extend to LiDAR blind zones → sky/rooftops provide gradients│
│     Ablation (Tab. VI): uncropped images achieve best calibration accuracy│
│                                                                          │
│  ❸ Rig optimization constraint effect                                     │
│     All frames share extrinsics → each update affects all frames → better generalization│
│     Ablation (Tab. V): remove rig → rotation error 0.13°→1.94° (15×)       │
│                                                                          │
│  ❹ "Stabilize first, release later" pose optimization strategy            │
│     Warmup 5 cycles → weight decay first half → cosine annealing → Refine freeze pose│
│     Each stage has clear design purpose, forming complete training strategy│
│                                                                          │
│  ❺ CUDA-level analytic pose gradients                                     │
│     Four independent paths computed in parallel on GPU → more efficient than autograd│
│     SE(3)/SO(3) operations in CUDA → geometric consistency guarantee      │
│                                                                          │
│  ❻ Practicality of adaptive voxel control                                 │
│     Different scene scales → different optimal anchor density               │
│     AVC auto-adapts → no manual tuning needed                             │
│                                                                          │
│  ❼ Scene-pose decoupled incremental accumulation                          │
│     Each iteration: step → SE3_exp → accumulate → reset                   │
│     Increment always near origin → small-angle approximation valid → more accurate gradients│
│                                                                          │
│  ❽ Extrinsic generalization = true sensor calibration                       │
│     Scene A optimization → Scene B/C transfer succeeds                      │
│     Proves TLC-Calib is not scene-overfitting                             │
│     But truly estimating physical sensor parameters                       │
│                                                                          │
│  ❾ GPU parallelism of Tile-Based rendering                                │
│     16×16 tile partitioning + prefix sum + radix sort → O(N log N) global sort│
│     Shared memory cooperative load + early stop → maximize GPU utilization│
│     Forward/backward symmetric design → backward reuses forward buffers   │
│                                                                          │
│  ❿ Most complete evaluation ecosystem                                     │
│     train.py → render.py → metrics_pose.py → metrics_nvs.py              │
│     → full_eval.py (batch aggregation) + Web Viewer (real-time visualization)│
│     Most complete open-source evaluation toolchain in 3DGS calibration    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

*Report generated: June 14, 2026 (ongoing updates)*  
*Coverage: All paper sections + all core code modules + CUDA forward/backward rendering + GPU memory management + camera model + Python binding layer + train.py comparison + 8 calibration methods cross-comparison*  
*Total chapters: 30 | Chinese version: docs/TLC-Calib_Paper_Analysis.md*
