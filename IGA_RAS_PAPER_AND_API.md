# IGA-Native Patch RAS 预条件子：原理、创新性与接口说明

> 适用场景：三维 NURBS/IGA H(curl) Maxwell-PML 问题  
> 实现文件：`covariant_aux_space/iga_patch_ras_preconditioner.{hpp,cpp}`  
> 接口头文件：`covariant_aux_space/covariant_preconditioner_factory.hpp`

---

## 目录

1. [问题背景](#1-问题背景)
2. [数学原理](#2-数学原理)
   - [IGA H(curl) 系统的实块表示](#21-iga-hcurl-系统的实块表示)
   - [NURBS 支撑 Patch 构造](#22-nurbs-支撑-patch-构造)
   - [受限加法 Schwarz（RAS）算子](#23-受限加法-schwarzras算子)
   - [Full-Rank 性质证明](#24-full-rank-性质证明)
   - [多 Schwarz 扫描](#25-多-schwarz-扫描)
3. [为何传统辅助空间方法在 PML 高阶 IGA 下失效](#3-为何传统辅助空间方法在-pml-高阶-iga-下失效)
4. [SCI 论文创新性总结](#4-sci-论文创新性总结)
5. [实现架构](#5-实现架构)
   - [Patch 构建算法](#51-patch-构建算法)
   - [局部矩阵装配策略](#52-局部矩阵装配策略)
   - [Mult 流程](#53-mult-流程)
6. [C++ 接口说明](#6-c-接口说明)
   - [直接构造](#61-直接构造)
   - [工厂接口（推荐）](#62-工厂接口推荐)
   - [参数说明](#63-参数说明)
7. [串行用法](#7-串行用法)
8. [MPI 并行用法](#8-mpi-并行用法)
9. [与 MFEM GMRES 的完整集成示例](#9-与-mfem-gmres-的完整集成示例)
10. [基准测试结果](#10-基准测试结果)
11. [局限性与未来工作](#11-局限性与未来工作)

---

## 1. 问题背景

考虑三维电磁波散射问题，在 PML（完美匹配层）截断计算域上离散化后得到复值
Maxwell 系统：

$$
\left(\frac{1}{\mu} \mathbf{C}^T \mathbf{M}_{\tilde{\Lambda}} \mathbf{C}
- \omega^2 \mathbf{M}_{\tilde{\varepsilon}}\right) \mathbf{u} = \mathbf{f},
$$

其中 $\tilde{\Lambda}$、$\tilde{\varepsilon}$ 是 PML 复拉伸张量，$\mathbf{C}$ 是 IGA
H(curl) curl 矩阵。用 NURBS 基函数离散（阶次 $p \geq 2$），得到一个大型复不定系统
$A_h \mathbf{u} = \mathbf{b}$，其中 $A_h \in \mathbb{C}^{N \times N}$。

**预条件 GMRES** 是目前求解此类问题的主流方法。AMS（HypreAMS）和辅助空间类方法
（`edge_yee`/`edge_galerkin`）在 PML 高阶 IGA 场景下均表现失效（500 次迭代真残差
仍 $\sim 10^{-3}$）。本文提出的 IGA-native 全秩 Schwarz 预条件子在同一问题上仅需
**20 次迭代**达到 $2.56 \times 10^{-7}$ 真相对残差。

---

## 2. 数学原理

### 2.1 IGA H(curl) 系统的实块表示

$A_h \in \mathbb{C}^{N \times N}$ 通过 MFEM 的 `HERMITIAN` convention 被表示为
$2N \times 2N$ 实块系统：

$$
\mathcal{A} =
\begin{pmatrix}
\operatorname{Re}(A_h) & -\operatorname{Im}(A_h) \\
\operatorname{Im}(A_h) &  \operatorname{Re}(A_h)
\end{pmatrix}
\in \mathbb{R}^{2N \times 2N}.
$$

预条件子在实块系统 $\mathcal{A} \mathbf{x} = \mathbf{b}$ 上工作，
其中 $\mathbf{x} = (\mathbf{u}^{\mathrm{re}}, \mathbf{u}^{\mathrm{im}})^T$。
这保留了 Re/Im 耦合结构，是 IGA-native RAS 的关键设计选择之一。

### 2.2 NURBS 支撑 Patch 构造

设 $\mathcal{T}_h$ 为 Bézier 元素集合，$V_h$ 为真自由度（true DOF）空间（扣除本质
边界条件和并行约束后）。

**Seed patch（零重叠层）**：对每个元素 $K \in \mathcal{T}_h$，令

$$
\mathcal{I}_K^{(0)} = \{ j \in V_h : \phi_j \text{ 在 } K \text{ 上不恒为零} \}
= \texttt{GetElementVDofs}(K) \cap \texttt{LocalTDof}\,,
$$

即以 `GetElementVDofs` 给出的 NURBS 基函数支撑集作为初始 patch DOF 集合。

**重叠扩展（$\ell$ 层）**：通过共享 true-DOF 的元素邻居图迭代扩展：

$$
\mathcal{E}_K^{(s+1)} = \mathcal{E}_K^{(s)} \cup
  \bigl\{ K' : \mathcal{I}_{K'}^{(0)} \cap \mathcal{I}_K^{(s)} \neq \emptyset \bigr\},
\quad
\mathcal{I}_K^{(s)} = \bigcup_{K' \in \mathcal{E}_K^{(s)}} \mathcal{I}_{K'}^{(0)}.
$$

**实块 Patch DOF**：考虑 $2N \times 2N$ 实块系统，patch $K$ 的完整索引集为：

$$
\hat{\mathcal{I}}_K = \mathcal{I}_K^{(\ell)} \cup \bigl(N + \mathcal{I}_K^{(\ell)}\bigr),
$$

将 Re 和 Im 部分同时包含在同一个局部问题中，保留 Re/Im 耦合。

### 2.3 受限加法 Schwarz（RAS）算子

设 $R_K \in \mathbb{R}^{|\hat{\mathcal{I}}_K| \times 2N}$ 为 patch $K$ 的限制算子，
局部矩阵为：

$$
\mathcal{A}_K = R_K \,\mathcal{A}\, R_K^T \in \mathbb{R}^{|\hat{\mathcal{I}}_K|
\times |\hat{\mathcal{I}}_K|}.
$$

设 $c_j$ 为 true-DOF $j$ 被覆盖的次数，定义对角权重矩阵 $W_K$ 使得

$$
W_K = \operatorname{diag}\!\left(\frac{1}{c_j}\right)_{j \in \hat{\mathcal{I}}_K},
\quad \sum_K R_K^T W_K R_K = I_{2N}.
$$

**单次 RAS 预条件应用**（$\omega$ 为阻尼系数）：

$$
\mathcal{B}^{-1} \mathbf{r}
= \omega \sum_{K} R_K^T W_K \mathcal{A}_K^{-1} R_K \mathbf{r}.
$$

**多次 Schwarz 扫描**（`iterations` = $s$）：

$$
\mathbf{z}_0 = \mathbf{0}, \quad
\boldsymbol{\rho}_k = \mathbf{r} - \mathcal{A}\, \mathbf{z}_k, \quad
\mathbf{z}_{k+1} = \mathbf{z}_k +
  \omega \sum_{K} R_K^T W_K \mathcal{A}_K^{-1} R_K \boldsymbol{\rho}_k.
$$

### 2.4 Full-Rank 性质证明

**命题**：假设

1. 每个 true-DOF $j$ 至少属于一个 patch（由代码中的 fallback patch `{j, N+j}` 保证）；
2. 权重 $w_j = 1/c_j > 0$（被覆盖次数有限）；
3. 每个局部矩阵 $\mathcal{A}_K$ 非奇异（稠密 LU + 对角正则化保证）。

则由 $\sum_K R_K^T W_K R_K = I$ 以及每个 $\mathcal{A}_K$ 可逆，可知
$\mathcal{B}^{-1}$ 的值域覆盖 $\mathbb{R}^{2N}$ 的所有方向。

这与辅助空间方法的关键区别：辅助空间校正 $\Pi \mathcal{A}_{\text{aux}}^{-1} \Pi^T$ 的
秩至多为 $\operatorname{rank}(\Pi)$（= 辅助空间维度），存在结构性缺秩。

### 2.5 多 Schwarz 扫描

当 `iterations` $> 1$ 时，每次扫描前重新计算当前残差 $\boldsymbol{\rho}_k$，
相当于在预条件子内部执行多步不精确 Richardson 迭代。这在单次全局 GMRES 迭代开销
允许时可提升预条件质量，但每次扫描调用一次 $\mathcal{A}\mathbf{z}$（集合通信，
对 MPI 安全）。

---

## 3. 为何传统辅助空间方法在 PML 高阶 IGA 下失效

辅助空间方法的核心假设是 **算子等价性**：辅助粗算子 $\mathcal{A}_{\text{aux}}$
与真实 Galerkin 算子 $\Pi^T \mathcal{A}_h \Pi$ 能量等价，即存在 $\alpha, \beta > 0$
使得

$$
\alpha\, (\Pi^T \mathcal{A}_h \Pi)\, \preceq\, \mathcal{A}_{\text{aux}}\,
\preceq\, \beta\, (\Pi^T \mathcal{A}_h \Pi).
$$

该假设在以下情况下失效：

| 失效原因 | 具体表现 |
|---|---|
| **PML 复拉伸张量** | NURBS H(curl) 在 PML 区域产生稠密长程 mass 耦合，Yee FDFD 5 点 stencil 无法近似 |
| **高阶 NURBS（$p \geq 3$）** | 更宽的基函数支撑导致更强的非局部耦合，Yee 格式误差随 $p$ 增大 |
| **单 patch IGA 几何** | 单 patch NURBS 网格使 IGA 质量矩阵全局稠密，Yee 局部 mass 近似精度更差 |

IGA-native RAS 完全绕过辅助空间假设：每个局部矩阵 $\mathcal{A}_K$ 从同一个装配好的
$\mathcal{A}$ 中直接提取，不依赖任何外部近似算子，PML 复拉伸已包含在内。

---

## 4. SCI 论文创新性总结

### Contribution 1：IGA 基函数支撑感知的 Patch 构造

传统 Schwarz 方法以几何域重叠或代数邻居图定义 patch。本工作利用
`GetElementVDofs()` 给出的 **NURBS 基函数支撑拓扑**定义 patch，使得：

- Patch 自然对齐 NURBS 高阶支撑宽度（阶次 $p$ 的 patch 宽度 ≈ $2(p+1)$ 个元素）；
- 零重叠层（`overlap_layers=0`）已产生足够大的 patch，避免了人工调参；
- patch 大小随 $p$ 自动增长，对高阶问题更鲁棒。

### Contribution 2：直接从组合 PML-IGA 算子提取局部矩阵

本工作定义 $\mathcal{A}_K = R_K \mathcal{A} R_K^T$ 中的 $\mathcal{A}$
是 **MFEM 装配后的真实 IGA-PML Galerkin 算子**，而非任何代理算子。

- 从 `ComplexHypreParMatrix`（MPI）或 `ComplexSparseMatrix`（串行）直接提取
  rank-local 对角块稀疏矩阵，无需全局 probing；
- 提取时间复杂度为 $O(\text{nnz})$，而 dense probing 为 $O(N^2)$；
- 局部矩阵包含 PML 的全部复拉伸信息，与全局求解器使用完全一致的算子。

### Contribution 3：Re/Im 完全耦合的局部求解

传统复 Maxwell 预条件子常对实部和虚部分别近似（块对角）。本工作在每个 patch 上
求解完整 $2|\mathcal{I}_K| \times 2|\mathcal{I}_K|$ 的实块系统，保留 Re/Im 耦合。
这对 PML 至关重要，因为 PML 算子的 $\operatorname{Re}$ 和 $\operatorname{Im}$
部分通过复拉伸张量强耦合。

### Contribution 4：MPI 并行下的 rank-local RAS

在分布式内存并行中，每个 MPI rank 在自己的局部 true-DOF 子集上独立构建 patch，
使用 `HypreParMatrix::GetDiag()` 提取 rank-local 对角块。所有 patch 构建和局部
求解完全本地计算，只有 GMRES 的全局内积和 $\mathcal{A} \mathbf{v}$ matvec 需要
跨 rank 通信。这是一种 **通信最小化** 的并行预条件子设计。

### Contribution 5：统一稀疏装配路径支持多种 MFEM 矩阵类型

实现了自动识别 `ComplexHypreParMatrix` / `HypreParMatrix` / `ComplexSparseMatrix`
/ `SparseMatrix` 四种矩阵类型的统一稀疏装配路径（`Assembly::Auto`），使得相同预条件
子代码对串行和 MPI 路径均适用，无需用户切换接口。

### 性能数字（可作为论文 Table）

```
硬 PML case: cube-nurbs.mesh, r=2, o=3 (IGA H(curl)), f=4.0 GHz, DOF=1344

方法                   外迭代  真相对残差      与 iga_ras 的迭代次数比
────────────────────────────────────────────────────────────────────────
无预条件子               500   3.997e-03      25×
Hypre AMS                500   8.341e-03      25×
edge_galerkin cps=2      500   1.746e-01      25×
edge_yee cps=2           500   1.982e-01      25×
iga_ras, overlap=0        20   2.559e-07       1×  ← 唯一收敛
```

---

## 5. 实现架构

### 5.1 Patch 构建算法

```
BuildElementPatches():
  1. 遍历本 rank 所有元素 e = 0..ne-1:
     - GetElementVDofs(e, vdofs)
     - 过滤 AbsDof(vdof) → GetLocalTDofNumber() >= 0 and < n
     - 得到 elem_tdofs[e] 和反向映射 dof_to_elems[tdof]

  2. 对每个 seed 元素 seed:
     - selected = {seed}
     - 循环 overlap_layers 次:
         next_selected = selected ∪ {e' : ∃e∈selected, ∃j∈elem_tdofs[e], e'∈dof_to_elems[j]}
         selected = next_selected

     - scalar_ids = ∪_{e∈selected} elem_tdofs[e]  (去重)
     - block_ids = scalar_ids ∪ {n+j : j∈scalar_ids}  (2x2 块索引)
     - coverage[j]++ for j in scalar_ids

  3. Fallback: 对每个 coverage[i]==0 的 DOF, 补加 patch {i, n+i}

  4. 权重: weights[i] = weights[n+i] = 1 / coverage[i]
```

**关键**：`GetElementVDofs` 对 NURBS 空间返回阶次 $p$ 下全支撑集，零重叠层已产生
宽度约 $(p+1)^3$ 的 patch（在 $r=2,o=3$ 的 cube 上，每 patch 约 600 个 DOF）。

### 5.2 局部矩阵装配策略

```
Auto 模式（推荐）:
  ComplexHypreParMatrix → GetSystemMatrix() → GetDiag() → A_local_ (rank-local CSR)
  HypreParMatrix        → GetDiag()          → A_local_
  ComplexSparseMatrix   → GetSystemMatrix()  → A_local_
  SparseMatrix          → 直接引用            → A_local_

BuildPatchSolverFromLocalSparse(ids):
  1. 构造 local_pos[j] = patch内位置 (O(n) 标记数组)
  2. 对 patch 内每行 row = ids[row_pos]:
     GetRow(row, cols, vals)
     对 cols 中在 patch 内的列 col_pos = local_pos[col]:
       block(row_pos, col_pos) = vals[q]
  3. 对角正则化: diag += max(1e-12, 1e-10 * max_diag)
  4. DenseMatrixInverse (dense LU)
```

稀疏提取时间复杂度：$O(\text{nnz\_local\_patch})$，远优于 dense probing 的
$O(N \cdot 2N)$。

### 5.3 Mult 流程

```cpp
Mult(r, z):
  z = 0, work = 0
  for it = 0..iterations-1:
      rho = r            // it=0: 初始残差
      if it > 0:
          rho = r - A * work  // 更新残差（MPI 集合操作）
      for each patch p:
          rhs[i] = rho[ids[i]]       // 局部提取
          sol = A_p^{-1} * rhs       // dense LU 局部求解
          work[ids[i]] += damping * weights[ids[i]] * sol[i]  // 加权累加
  z = work
```

**MPI 安全性**：
- `it=0` 时无跨 rank 通信；
- `it>0` 时 `A_.Mult(work_, Awork_)` 是 `HypreParMatrix` 的集合 matvec，
  但在所有 rank 上同步调用（均处于 `Mult()` 内部），不存在 collective 不匹配风险；
- Patch 局部求解完全本地，权重向量本地存储。

---

## 6. C++ 接口说明

```cpp
#include "covariant_aux_space/iga_patch_ras_preconditioner.hpp"
#include "covariant_aux_space/covariant_preconditioner_factory.hpp"
```

### 6.1 直接构造

```cpp
namespace covariant_aux_space {

class IGAPatchRASPreconditioner : public mfem::Solver
{
public:
    struct Options
    {
        enum class Assembly { Auto, LocalSparse, DenseProbe };

        int         overlap_layers = 0;    // NURBS 支撑 patch 重叠层数
        int         iterations     = 1;    // 每次应用内的 Schwarz 扫描次数
        mfem::real_t damping       = 1.0;  // RAS 加权阻尼系数 ω
        bool        verbose        = true; // 构建时打印诊断信息
        Assembly    assembly       = Assembly::Auto;
    };

    // 主构造函数
    IGAPatchRASPreconditioner(const mfem::Operator &A,
                              const mfem::ParFiniteElementSpace &fespace,
                              const Options &options);

    // 便捷构造函数（参数顺序版）
    IGAPatchRASPreconditioner(const mfem::Operator &A,
                              const mfem::ParFiniteElementSpace &fespace,
                              int overlap_layers,
                              mfem::real_t damping,
                              int iterations);

    void Mult(const mfem::Vector &r, mfem::Vector &z) const override;

    int  NumPatches()   const;  // 返回总 patch 数
    int  MaxPatchSize() const;  // 返回最大 patch 维度（2x2 块）
};
```

### 6.2 工厂接口（推荐）

```cpp
namespace covariant_aux_space {

struct IGAPatchRASConfig
{
    int    overlap_layers = 0;
    double damping        = 0.8;
    int    iterations     = 1;
    bool   verbose        = true;
    IGAPatchRASPreconditioner::Options::Assembly assembly =
        IGAPatchRASPreconditioner::Options::Assembly::Auto;

    // GMRES 参数（用于 CreatePreconditionedGMRES）
    int    gmres_max_iter = 500;
    int    gmres_print    = 0;
    double gmres_rel_tol  = 1e-5;
    double gmres_abs_tol  = 0.0;
};

// 工厂：创建预条件子
std::unique_ptr<IGAPatchRASPreconditioner>
CreateIGAPatchRASPreconditioner(
    const mfem::Operator &A,
    const mfem::ParFiniteElementSpace &fespace,
    const IGAPatchRASConfig &cfg = IGAPatchRASConfig{});

// 工厂：创建预条件 GMRES
mfem::GMRESSolver
CreatePreconditionedGMRES(
    const mfem::Operator &A,
    IGAPatchRASPreconditioner &prec,
    const IGAPatchRASConfig &cfg = IGAPatchRASConfig{});
}
```

### 6.3 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `overlap_layers` | `int` | `0` | **最重要参数**。0 = NURBS 支撑 patch，已足够大；1 = 扩展一层元素邻居；≥2 在小 mesh 上可能接近全局 |
| `damping` | `real_t` | `1.0`（Options）/ `0.8`（Config）| RAS 修正阻尼 $\omega$。0.8–1.0 通常最优 |
| `iterations` | `int` | `1` | 内层 Schwarz 扫描次数。增加 1 次扫描对 MPI 路径多一次 AllReduce |
| `assembly` | `Assembly` | `Auto` | `Auto`：自动选择稀疏提取（推荐）；`LocalSparse`：强制稀疏；`DenseProbe`：强制 dense（仅调试） |
| `verbose` | `bool` | `true` | 构建完成后 rank 0 打印 patches/max_dim/assembly 信息 |

**`overlap_layers` 对 patch 尺寸的影响**（`cube-nurbs.mesh, r=2, o=3`）：

| `overlap_layers` | patches | `max_patch_dim` | GMRES iters | 备注 |
|---:|---:|---:|---:|---|
| 0 | 64 | 600 | **20** | 推荐，scalable |
| 1 | 64 | 2688 | 1 | 接近全局；上界诊断用 |

---

## 7. 串行用法

### 最简示例

```cpp
#include "mfem.hpp"
#include "covariant_aux_space/iga_patch_ras_preconditioner.hpp"

using namespace mfem;

// 假设已有：
//   ParFiniteElementSpace *fespace  （NURBS H(curl)）
//   Operator *A                     （ComplexSparseMatrix 或 ComplexHypreParMatrix 含 np=1）
//   Vector B, X                     （右端向量，初始解）

covariant_aux_space::IGAPatchRASPreconditioner::Options opts;
opts.overlap_layers = 0;
opts.damping        = 0.8;
opts.iterations     = 1;
opts.assembly       = covariant_aux_space::IGAPatchRASPreconditioner::Options::Assembly::Auto;

covariant_aux_space::IGAPatchRASPreconditioner prec(*A, *fespace, opts);

GMRESSolver gmres(MPI_COMM_WORLD);  // np=1 时等价于串行
gmres.SetOperator(*A);
gmres.SetPreconditioner(prec);
gmres.SetKDim(200);
gmres.SetMaxIter(500);
gmres.SetRelTol(1e-7);
gmres.SetAbsTol(0.0);
gmres.Mult(B, X);

std::cout << "iters=" << gmres.GetNumIterations()
          << ", converged=" << gmres.GetConverged() << std::endl;
```

### 工厂方式（串行）

```cpp
#include "covariant_aux_space/covariant_preconditioner_factory.hpp"

covariant_aux_space::IGAPatchRASConfig cfg;
cfg.overlap_layers = 0;
cfg.damping        = 0.8;
cfg.iterations     = 1;
cfg.gmres_rel_tol  = 1e-7;
cfg.gmres_max_iter = 500;

auto prec = covariant_aux_space::CreateIGAPatchRASPreconditioner(*A, *fespace, cfg);
auto gmres = covariant_aux_space::CreatePreconditionedGMRES(*A, *prec, cfg);
gmres.Mult(B, X);
```

### 带真残差早停（串行，推荐用于 benchmarking）

```cpp
// pml_point_source_demo 中的 TrueResidualController 已集成，命令行触发：
// -trc -trp 5 -grt 1e-7
```

---

## 8. MPI 并行用法

### 设计原则

**rank-local RAS**：每个 MPI rank 仅在自己 own 的 true-DOF 上构建 patch，使用
`HypreParMatrix::GetDiag()` 提取 rank-local 对角稀疏块。没有任何 patch 会跨越 MPI
partition 边界（`GetLocalTDofNumber() >= 0 && < n` 过滤掉 ghost DOF）。

通信点：
- GMRES 内积 → `MPI_Allreduce`（MFEM 内部）
- `A_.Mult()` matvec → HypreParMatrix ghost 通信（MFEM 内部）
- 预条件子 `Mult()` 内 `iterations>1` 时的 `A_.Mult()` → 同上

**预条件子本身不需要额外 MPI 通信**，所有 patch 构建和 patch 求解完全本地。

### MPI 示例

```cpp
#include "mfem.hpp"
#include "covariant_aux_space/iga_patch_ras_preconditioner.hpp"

// MPI 初始化由 MFEM Mpi 类管理
mfem::Mpi::Init(argc, argv);
mfem::Hypre::Init();

// ... 网格分区、FESpace 构建、ComplexSesquilinearForm 装配 ...
// 得到：
//   ParFiniteElementSpace *fespace（分布式）
//   Operator *A（ComplexHypreParMatrix，分布式 2N×2N 实块系统）
//   Vector B, X（rank-local 分量）

covariant_aux_space::IGAPatchRASPreconditioner::Options opts;
opts.overlap_layers = 0;    // rank-local NURBS patch
opts.damping        = 0.8;
opts.iterations     = 1;
// opts.assembly = Auto 自动选 ComplexHypreParMatrix → GetSystemMatrix() → GetDiag()

covariant_aux_space::IGAPatchRASPreconditioner prec(*A, *fespace, opts);

GMRESSolver gmres(MPI_COMM_WORLD);
gmres.SetOperator(*A);
gmres.SetPreconditioner(prec);
gmres.SetKDim(200);
gmres.SetMaxIter(20);
gmres.SetRelTol(1e-5);
gmres.Mult(B, X);

if (mfem::Mpi::Root())
{
    std::cout << "np=" << mfem::Mpi::WorldSize()
              << " iters=" << gmres.GetNumIterations()
              << " converged=" << gmres.GetConverged() << std::endl;
}
```

### 带 MPI-safe 真残差早停

```cpp
// TrueResidualController::GlobalNorml2 通过 MPI_Allreduce 计算全局 L2 范数
// 在 pml_point_source_demo 中命令行直接使用：
// mpirun -np 2 ./pml_point_source_demo -prec iga_ras -trc -trp 2 ...

// 若在自己的代码中使用 TrueResidualController，确保传入 MPI_COMM_WORLD：
TrueResidualController ctrl(*A, B, bnorm, rel_tol, 0.0, print_every,
                            mfem::Mpi::WorldRank(), MPI_COMM_WORLD);
gmres.SetController(ctrl);
```

### 命令行（`pml_point_source_demo`）

```bash
# MPI np=2 完整示例
export OPAL_PREFIX=/mnt/f/optemcode/opt/openmpi
export LD_LIBRARY_PATH=$OPAL_PREFIX/lib:/mnt/f/optemcode/opt/hypre/lib

$OPAL_PREFIX/bin/mpirun -np 2 ./pml_point_source_demo \
  -m /mnt/f/optemcode/mfem/data/cube-nurbs.mesh \
  -r 1 -o 2 -f 4.0 \
  -prec iga_ras \
  -rasov 0 \        # overlap_layers
  -rasw 0.8 \       # damping
  -rasit 1 \        # iterations
  -rasasm auto \    # assembly mode
  -trc -trp 2 \     # 真残差监控（MPI-safe）
  -gmi 20 -gpl 0 -grt 1e-5 \
  -no-vis

# 等价的 smoke test 脚本
bash run_iga_ras_mpi_smoke.sh
```

### MPI smoke test 输出参考

```
[iga_ras] patches=4 per rank, overlap_layers=0, max_patch_dim=288,
          damping=0.8, iterations=1, assembly=local_sparse

[PML GMRES] iga_ras start ||r0||=4.3056 (rel=1)
[PML GMRES] true residual controller enabled (rel_tol=1e-05)
[true-residual monitor] iter=0 ||r||=4.3056 (rel=1),         true_converged=0
[true-residual monitor] iter=2 ||r||=0.02089 (rel=4.85e-3),  true_converged=0
[true-residual monitor] iter=4 ||r||=2.22e-3 (rel=5.16e-4),  true_converged=0
[true-residual monitor] iter=6 ||r||=1.94e-4 (rel=4.51e-5),  true_converged=0
[true-residual monitor] iter=8 ||r||=4.85e-6 (rel=1.13e-6),  true_converged=1
[PML GMRES] iga_ras done  iters=8, converged=1, true_converged=1
```

---

## 9. 与 MFEM GMRES 的完整集成示例

下面是一个自包含的最小 MFEM IGA H(curl) PML 求解器，展示预条件子从装配到求解的
完整流程：

```cpp
// minimal_iga_ras_solver.cpp
#include "mfem.hpp"
#include "covariant_aux_space/iga_patch_ras_preconditioner.hpp"
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   // 1. 网格
   const char *mesh_file = "cube-nurbs.mesh";
   int ref = 1, order = 2;
   Mesh serial_mesh(mesh_file);
   for (int i = 0; i < ref; i++) serial_mesh.UniformRefinement();
   auto pmesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, serial_mesh);

   // 2. FESpace
   FiniteElementCollection *fec = new NURBS_HCurlFECollection(order, 3);
   auto fespace = std::make_unique<ParFiniteElementSpace>(pmesh.get(), fec);

   // 3. 装配（略）—— 得到 ComplexParSesquilinearForm a
   //    a.FormSystemMatrix(ess_tdof_list, ess_tdof_list, A);
   //    A 类型为 OperatorPtr，内部是 ComplexHypreParMatrix
   OperatorPtr A;
   Vector B(fespace->GetTrueVSize() * 2), X(fespace->GetTrueVSize() * 2);
   // ...（填充 A, B）

   // 4. 预条件子
   covariant_aux_space::IGAPatchRASPreconditioner::Options opts;
   opts.overlap_layers = 0;
   opts.damping        = 0.8;
   opts.iterations     = 1;
   covariant_aux_space::IGAPatchRASPreconditioner prec(*A, *fespace, opts);

   // 5. GMRES
   GMRESSolver gmres(MPI_COMM_WORLD);
   gmres.SetOperator(*A);
   gmres.SetPreconditioner(prec);
   gmres.SetKDim(200);
   gmres.SetMaxIter(500);
   gmres.SetRelTol(1e-7);
   gmres.Mult(B, X);

   if (Mpi::Root())
      cout << "iters=" << gmres.GetNumIterations()
           << " converged=" << gmres.GetConverged() << endl;
   return 0;
}
```

---

## 10. 基准测试结果

### 硬 PML case（SCI 论文 Table 主体）

**Setup**：`cube-nurbs.mesh`，均匀细化 2 次，NURBS H(curl) 阶次 3，频率 4.0 GHz，
PML 厚度 0.25，真 DOF = 1344（实块 2688）

| 方法 | 外层 GMRES 迭代 | 最终真相对残差 | 最终 GMRES 相对残差 |
|---|---:|---:|---:|
| 无预条件 | 500 | 3.997e-03 | — |
| Hypre AMS | 500 | 8.341e-03 | — |
| edge_galerkin cps=2 | 500 | 1.746e-01 | — |
| edge_yee cps=2 (pure FDFD) | 500 | 1.982e-01 | — |
| **iga_ras overlap=0** | **20** | **2.559e-07** | **<1e-7** |

```bash
# 重现命令
./pml_point_source_demo \
  -m /mnt/f/optemcode/mfem/data/cube-nurbs.mesh \
  -r 2 -o 3 -f 4.0 \
  -prec iga_ras -rasov 0 -rasw 0.8 -rasit 1 \
  -trc -gmi 500 -gpl 0 -grt 1e-7 -no-vis
```

### MPI 扩展性（初步）

| np | r | o | iters | true rel |
|---:|---:|---:|---:|---:|
| 1 | 2 | 3 | 20 | 2.56e-07 |
| 2 | 1 | 2 | 8  | 1.13e-06 |

> 注：`np=2` 的 case 规模小于 `np=1`（r=1 vs r=2），非直接可比的扩展性数据。
> 跨 rank overlap 未实现，对大 mesh 的 strong scaling 待补充。

---

## 11. 局限性与未来工作

| 限制 | 当前状态 | 计划 |
|---|---|---|
| **Patch 局部求解为 dense LU** | `r=2,o=3,ov=0` 时 max_patch=600，可接受；`ov=1` 时 max_patch=2688，内存压力大 | 替换为稀疏 ILU 或 sparse direct（UMFPACK/SuperLU）|
| **跨 MPI rank overlap 未实现** | `ov≥1` 在多 rank 时只扩展本 rank 内的邻居 | 需要 ghost layer 共享；使用 `ParFiniteElementSpace` 的 shared DOF 机制 |
| **单 patch 网格限制** | `SinglePatchNURBSEvaluator` 仅支持单 patch；multi-patch IGA 未测试 | `iga_ras` 本身无此限制；multi-patch 待网格适配 |
| **FGMRES 下的内层 A.Mult()** | `iterations>1` 时调用 A.Mult()，使预条件子变为 variable preconditioner | 若与 FGMRES 配合使用可能需要显式切换；普通 GMRES 无此问题 |
| **强扩展性数据** | 仅有 np=2 smoke test | 需 np=1/2/4/8 统一 mesh 扩展性 benchmark |
