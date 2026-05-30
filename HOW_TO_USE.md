# MFEM IGA H(curl) Maxwell 求解器预条件子使用手册

> 版本：2026-05-28  
> 适用场景：三维 NURBS/IGA H(curl) Maxwell 问题（cavity、PML 吸收边界）  
> 主要参考代码：`pml_point_source_demo.cpp`、`covariant_aux_space/`

---

## 目录

1. [快速决策指南](#1-快速决策指南)
2. [编译](#2-编译)
3. [C++ API 用法](#3-c-api-用法)
   - [iga_ras（推荐主路线）](#31-iga_ras推荐主路线)
   - [edge_galerkin](#32-edge_galerkin)
   - [edge_yee](#33-edge_yee)
4. [命令行接口（pml_point_source_demo）](#4-命令行接口pml_point_source_demo)
5. [真残差监控（-trc，MPI 安全）](#5-真残差监控-trc-mpi-安全)
6. [MPI 并行用法](#6-mpi-并行用法)
7. [基准测试参考数据](#7-基准测试参考数据)
8. [已知限制](#8-已知限制)

---

## 1. 快速决策指南

```
你的问题是哪种？
│
├─ Cavity（无 PML），order ≤ 2
│    └─→ edge_yee  -ka -cps 1          ← 最快，12% aux DOF，13 iter
│
├─ PML，order ≤ 2
│    └─→ edge_yee  -ka -cps 1          ← 43 iter (pure FDFD) / 8 iter (Galerkin fallback)
│
└─ PML，order = 3  或  需要可靠真残差收敛
     └─→ iga_ras  -rasov 0 -rasw 0.8   ← 20 iter，true rel 2.6e-7（硬 case 首选）
```

| 方法 | 核心思路 | 适用 | 不适用 |
|---|---|---|---|
| **iga_ras** | 从真实 IGA 算子提取 NURBS 支撑 patch，full-rank RAS | PML 高阶、可靠 true residual | 极大 mesh（patch LU 尚未换 ILU） |
| **edge_galerkin** | 辅助 Yee 边空间 + Π^T A_h Π | 诊断、验证 transfer 质量 | 无 smoother 时高阶 PML 不收敛 |
| **edge_yee** | 辅助 Yee 边空间 + 独立 FDFD 算子 | Cavity 低阶、快速 setup | PML order=3 结构性失败 |

---

## 2. 编译

```bash
cd /mnt/f/optemcode/opt_em_iga_repo
WORKSPACE_DIR=/mnt/f/optemcode make pml_point_source_demo
```

运行时库路径：

```bash
export LD_LIBRARY_PATH=/mnt/f/optemcode/opt/openmpi/lib:/mnt/f/optemcode/opt/hypre/lib
```

---

## 3. C++ API 用法

公共入口头文件：

```cpp
#include "covariant_aux_space/covariant_preconditioner_factory.hpp"
```

### 3.1 `iga_ras`（推荐主路线）

#### 最简用法

```cpp
#include "covariant_aux_space/iga_patch_ras_preconditioner.hpp"

// A       : 已装配的复 Maxwell 系统算子（HypreParMatrix 包装的 2N×2N 实块系统）
// fespace : ParFiniteElementSpace（NURBS H(curl)）
covariant_aux_space::IGAPatchRASPreconditioner::Options opts;
opts.overlap_layers = 0;     // 0 = NURBS 支撑元素 patch（推荐起点）
opts.damping        = 0.8;   // RAS 加权阻尼
opts.iterations     = 1;     // 每次预条件应用内 RAS 扫描次数

covariant_aux_space::IGAPatchRASPreconditioner prec(A, fespace, opts);

mfem::GMRESSolver gmres(MPI_COMM_WORLD);
gmres.SetOperator(A);
gmres.SetPreconditioner(prec);
gmres.SetMaxIter(500);
gmres.SetRelTol(1e-7);
gmres.Mult(B, X);
```

#### 工厂方式（推荐用于外部 demo）

```cpp
#include "covariant_aux_space/covariant_preconditioner_factory.hpp"

covariant_aux_space::IGAPatchRASConfig cfg;
cfg.overlap_layers = 0;
cfg.damping        = 0.8;
cfg.iterations     = 1;
cfg.gmres_rel_tol  = 1e-7;
cfg.gmres_max_iter = 500;

auto prec = covariant_aux_space::CreateIGAPatchRASPreconditioner(A, fespace, cfg);
auto gmres = covariant_aux_space::CreatePreconditionedGMRES(A, *prec, cfg);
gmres.Mult(B, X);
```

#### `Options::Assembly` 后端选择

```cpp
// 默认：Auto（推荐）——从 ComplexHypreParMatrix / HypreParMatrix /
//        ComplexSparseMatrix / SparseMatrix 抽取 rank-local CSR 块
opts.assembly = IGAPatchRASPreconditioner::Options::Assembly::Auto;

// 仅当算子类型不受支持或调试时使用：
opts.assembly = IGAPatchRASPreconditioner::Options::Assembly::DenseProbe;
```

#### `overlap_layers` 如何选

| `overlap_layers` | 含义 | 适用场景 |
|---:|---|---|
| 0 | 每个 knot element 的 NURBS 支撑 DOF patch | **默认，scalable** |
| 1 | 通过共享 DOF 邻居图扩展一层 | 调参诊断，小 mesh 可用 |
| ≥2 | 在小 mesh 上可能扩展到全系统 | 上界诊断，不推荐用于生产 |

---

### 3.2 `edge_galerkin`

```cpp
#include "covariant_aux_space/covariant_preconditioner_factory.hpp"
#include "fdfd_iga_init/reference_patch_evaluator.hpp"

// 需要先构造 NURBS 几何评估器
fdfd_iga_init::SinglePatchNURBSEvaluator geom(*pmesh, *pmesh->NURBSext, 0);
std::function<double(const mfem::Vector &)> eps_fn = [](const auto &) { return 1.0; };

covariant_aux_space::PreconditionerConfig cfg;
cfg.mode           = "edge_galerkin";
cfg.knot_align     = true;
cfg.cells_per_span = 1;       // 每个 knot span 一个 Yee cell
cfg.wave_number    = k0;      // k0 = omega * sqrt(mu * eps)

auto prec = covariant_aux_space::CreateCovariantPreconditioner(fespace, geom, eps_fn, cfg);
prec->SetOperator(A);

mfem::GMRESSolver gmres(MPI_COMM_WORLD);
gmres.SetOperator(A);
gmres.SetPreconditioner(*prec);
gmres.Mult(B, X);
```

---

### 3.3 `edge_yee`

```cpp
covariant_aux_space::PreconditionerConfig cfg;
cfg.mode           = "edge_yee";
cfg.knot_align     = true;
cfg.cells_per_span = 1;
cfg.wave_number    = k0;

// PML 网格：保持默认 yee_pml=false（不开 Yee PML 拉伸），
// 则 PML 区域自动 fallback 到 Π^T A_h Π（推荐）
cfg.yee_pml        = false;
cfg.no_pml_fallback = false;

// Cavity 网格：无 PML，直接用 Yee FDFD
// cfg.yee_pml = false; cfg.no_pml_fallback = true;

auto prec = covariant_aux_space::CreateCovariantPreconditioner(fespace, geom, eps_fn, cfg);
prec->SetOperator(A);
```

---

## 4. 命令行接口（`pml_point_source_demo`）

```
./pml_point_source_demo [options]
```

### 核心选项

| 选项 | 默认值 | 说明 |
|---|---|---|
| `-m <mesh>` | `cube-nurbs.mesh` | NURBS 网格文件 |
| `-r <n>` | 1 | 均匀细化次数 |
| `-o <n>` | 1 | NURBS 阶次 |
| `-f <Hz>` | 5.0 | 激励频率 |
| `-prec <name>` | `none` | 预条件子：`none` / `ams` / `iga_ras` / `edge_galerkin` / `edge_yee` |
| `-gmi <n>` | 500 | GMRES 最大迭代数 |
| `-grt <tol>` | 1e-5 | GMRES 相对收敛容差 |
| `-gpl <n>` | 1 | GMRES 打印级别（0=静默） |
| `-no-vis` | — | 关闭 GLVis 可视化 |

### `iga_ras` 专属选项

| 选项 | 默认值 | 说明 |
|---|---|---|
| `-rasov <n>` | 0 | 元素邻居重叠层数 |
| `-rasw <w>` | 1.0 | patch correction 阻尼权重 |
| `-rasit <k>` | 1 | 每次预条件应用内扫描次数 |
| `-rasasm <s>` | `auto` | 装配模式：`auto` / `local_sparse` / `dense_probe` |

### `edge_yee` / `edge_galerkin` 专属选项

| 选项 | 默认值 | 说明 |
|---|---|---|
| `-ka` / `-no-ka` | on | 开启 knot-span 对齐 Yee 网格 |
| `-cps <n>` | 1 | 每个 knot span 的 Yee 格数 |
| `-an <n>` | 5 | 非 knot-align 时每方向 aux 节点数 |
| `-npf` / `-no-npf` | off | 禁用 PML Galerkin fallback（强制纯 Yee） |

### 真残差控制

| 选项 | 默认值 | 说明 |
|---|---|---|
| `-trc` / `-no-trc` | off | 按 **真（非预条件）残差** 早停，MPI 下已修复为 collective-safe |
| `-trp <n>` | 0 | 每 n 次迭代打印一次真残差（0=只打印最终） |
| `-grt <tol>` | 1e-5 | 与 `-trc` 配合的真残差相对容差 |

---

## 5. 真残差监控（`-trc`，MPI 安全）

### 背景

MFEM 的 GMRES 默认使用**预条件残差** `||P⁻¹(b - Ax)||` 作为收敛判断。对于 deflation
型预条件子（如 `edge_yee`），预条件残差可以很小但真残差仍然较大。`-trc` 让 GMRES 按
**未预条件真残差** `||b - Ax||` 早停。

### MPI 修复（2026-05-28）

旧实现在 MPI 下调用 `v.Norml2()` 得到 rank-local 范数，各 rank 判断结果不一致导致
hang。已修复为通过 `MPI_Allreduce` 计算全局 L2 范数：

```cpp
// 内部实现（pml_point_source_demo.cpp 中的静态方法）
static real_t GlobalNorml2(const Vector &v, MPI_Comm comm)
{
   double local_sq = static_cast<double>(v.Norml2());
   local_sq *= local_sq;
   double global_sq = 0.0;
   MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, comm);
   return static_cast<real_t>(std::sqrt(global_sq));
}
```

所有 rank 在每次 callback 中同步计算全局范数，`converged` 标志在所有 rank 上一致。

### 推荐用法

```bash
# 串行：hard PML case，真残差目标 1e-7，每 5 步打印
./pml_point_source_demo \
  -m /mnt/f/optemcode/mfem/data/cube-nurbs.mesh \
  -r 2 -o 3 -f 4.0 \
  -prec iga_ras -rasov 0 -rasw 0.8 -rasit 1 \
  -trc -trp 5 -gmi 500 -gpl 0 -grt 1e-7 -no-vis

# MPI：同样命令，-trc 现在可安全使用
OPAL_PREFIX=/mnt/f/optemcode/opt/openmpi \
  $OPAL_PREFIX/bin/mpirun -np 2 ./pml_point_source_demo \
  -m /mnt/f/optemcode/mfem/data/cube-nurbs.mesh \
  -r 1 -o 2 -f 4.0 \
  -prec iga_ras -rasov 0 -rasw 0.8 -rasit 1 \
  -trc -trp 2 -gmi 20 -gpl 0 -grt 1e-5 -no-vis
```

### 输出示例（MPI np=2，r=1，o=2）

```
[PML GMRES] iga_ras start ||r0||=4.3056 (rel=1)
[PML GMRES] true residual controller enabled (rel_tol=1e-05)
[true-residual monitor] iter=0  ||r||=4.3056   (rel=1),         true_converged=0
[true-residual monitor] iter=2  ||r||=0.020885 (rel=4.85e-03),  true_converged=0
[true-residual monitor] iter=4  ||r||=0.002223 (rel=5.16e-04),  true_converged=0
[true-residual monitor] iter=6  ||r||=1.94e-04 (rel=4.51e-05),  true_converged=0
[true-residual monitor] iter=8  ||r||=4.85e-06 (rel=1.13e-06),  true_converged=1
[PML GMRES] iga_ras done  iters=8, converged=1, true_converged=1
```

---

## 6. MPI 并行用法

```bash
# 使用项目自带 MPI smoke test 脚本
bash run_iga_ras_mpi_smoke.sh

# 或手动指定
export OPAL_PREFIX=/mnt/f/optemcode/opt/openmpi
export LD_LIBRARY_PATH=$OPAL_PREFIX/lib:/mnt/f/optemcode/opt/hypre/lib

$OPAL_PREFIX/bin/mpirun -np 2 ./pml_point_source_demo \
  -m /mnt/f/optemcode/mfem/data/cube-nurbs.mesh \
  -r 1 -o 2 -f 4.0 \
  -prec iga_ras -rasov 0 -rasw 0.8 -rasit 1 -rasasm auto \
  -trc -gmi 20 -gpl 0 -grt 1e-5 -no-vis
```

### 三种预条件子的 MPI 正确性

| 方法 | MPI 下不 hang | 数学正确 | 说明 |
|---|:---:|:---:|---|
| **`iga_ras`** | ✅ | ✅ | rank-local RAS，合法并行预条件子，已 np=2 验证 |
| **`edge_yee`** | ⚠️ | ❌ | `Pi_` 仅含 rank-local DOF；投影丢失其他 rank 的贡献；退化为局部校正，预条件效果下降 |
| **`edge_galerkin`** | ⚠️ | ❌ | 同 edge_yee，另 `Pi^T A_h Pi` 粗矩阵也只用本 rank 的行，粗算子不完整 |

> **结论：MPI 并行仿真只应使用 `iga_ras`。**  
> `edge_yee`/`edge_galerkin` 在 MPI 下不会 crash，但投影丢失跨 rank 贡献，  
> 数学上不是全局辅助空间校正，收敛可能大幅劣化。若需并行辅助空间方法，  
> 须为 `Pi_` 构建加 `MPI_Allreduce` 收集全局投影（待实现）。

### `iga_ras` MPI 功能状态

| 功能 | 状态 |
|---|---|
| rank-local patch 构建 | ✅ 已实现 |
| rank-local CSR 块提取（Auto/LocalSparse） | ✅ 已实现 |
| `-trc` 真残差早停 | ✅ 已修复为 MPI collective-safe |
| 跨 MPI rank overlap | ⏳ 待实现（下一优先级） |
| patch local sparse ILU/direct | ⏳ 待实现 |

---

## 7. 基准测试参考数据

### 硬 PML 测试场景

**网格**：`cube-nurbs.mesh`，`r=2`，`o=3`，`f=4.0`，true DOF = 1344（实块 2688）

| 预条件子 | 外层迭代 | 真相对残差 | 备注 |
|---|---:|---:|---|
| none | 500 | 3.997e-03 | 不收敛 |
| AMS（Hypre） | 500 | 8.341e-03 | NURBS 上结构性失败 |
| edge_galerkin cps=2 | 500 | 1.746e-01 | 无 smoother 时不够强 |
| edge_yee cps=2（纯 FDFD） | 500 | 1.982e-01 | p=3 PML 算子等价性失败 |
| **iga_ras overlap=0** | **20** | **2.559e-07** | ✅ 超过 1e-5 目标 |

运行命令：

```bash
./pml_point_source_demo \
  -m /mnt/f/optemcode/mfem/data/cube-nurbs.mesh \
  -r 2 -o 3 -f 4.0 \
  -prec iga_ras -rasov 0 -rasw 0.8 -rasit 1 \
  -trc -gmi 500 -gpl 0 -grt 1e-7 -no-vis
```

### MPI smoke test（np=2，r=1，o=2）

```
assembly=local_sparse, patches per rank ≈ 4, max_patch_dim=288
iters=8, true rel=1.127e-06   (目标 1e-5，-trc 早停触发)
```

### 完整对比 benchmark

```bash
bash run_full_preconditioner_benchmark.sh
# 输出到 benchmark_results/full_preconditioners_<timestamp>/
```

---

## 8. 已知限制

| 限制 | 说明 | 计划 |
|---|---|---|
| **跨 MPI rank overlap 未实现** | 当前 RAS 只在 rank 内建 patch；`rasov≥1` 在大 mesh 上可跨 rank，但目前 ghost DOF 未纳入 | 下一优先级 |
| **Patch 求解仍为 dense LU** | 大 patch（`rasov≥1` 在小 mesh 上） 内存可达全系统；`r=2,o=3` 时 `rasov=1` max_patch_dim=2688 | 待加 `-rasloc sparse_ilu\|sparse_direct` |
| **`edge_yee` 在 PML p=3 失败** | p=3 时 NURBS H(curl) PML 产生 dense long-range mass coupling，FDFD 5-point stencil 无法近似 | 报告为 limitation；用 `iga_ras` 代替 |
| **非均匀 knot vector 奇异** | `nonuniform7-cube-nurbs.mesh` 在 Yee FDFD 装配时奇异 | 近期 workaround：用均匀 knot + CP 位置弯曲 |
| **`fdfd_iga_init_demo` FDFD 初值不可用** | 旧 cavity 初值 demo 在当前 tree 下 timeout/abort | 待单独修复后加入 PML benchmark 表 |
| **单 patch 限制** | `SinglePatchNURBSEvaluator` 限定单 patch；`edge_yee`/`edge_galerkin` 不支持多 patch | `iga_ras` 无此限制 |

---

## 附：文件索引

```
covariant_aux_space/
  iga_patch_ras_preconditioner.hpp/.cpp   ← IGA-native RAS（主路线）
  covariant_reference_preconditioner.hpp/.cpp  ← edge_yee / edge_galerkin
  covariant_preconditioner_factory.hpp    ← 统一工厂入口（外部 demo 首选）
  yee_transfer.hpp/.cpp                  ← fast-Π transfer 构建
  yee_operator.hpp/.cpp                  ← Yee FDFD 算子
  MFEM_IGA_HCURL_PRECONDITIONERS.md      ← API 参考（英文，含 dense probe 细节）
  IGA_NATIVE_SCHWARZ_PRECONDITIONER.md   ← RAS 理论推导

pml_point_source_demo.cpp               ← PML 驱动（支持所有预条件子）
run_iga_ras_benchmark.sh                ← IGA RAS 专项 benchmark
run_full_preconditioner_benchmark.sh    ← 全预条件子对比 benchmark
run_iga_ras_mpi_smoke.sh                ← MPI smoke test
PROGRESS.md                             ← 完整开发日志（含所有实验数据）
```
