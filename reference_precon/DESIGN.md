# Reference-Domain Preconditioning for IGA H(curl) Maxwell Solvers
# 参考域预处理：借助 Yee 网格加速 IGA 迭代求解

---

## 核心观察

IGA 的参数域（reference domain）$\hat{\Omega} = [0,1]^3$ 是一个**规则正方形/立方体**。
物理域 $\Omega$ 通过 NURBS 几何映射 $F: \hat{\Omega} \to \Omega$ 得到。

$$
\text{物理域 Maxwell (IGA)} \xleftrightarrow{\text{Piola 变换}} \text{参数域 Maxwell (变系数 Yee)}
$$

这意味着 maxwellb 风格的矩阵-向量积可以在**参数空间**直接实现——无需组装稀疏矩阵，
利用规则格点的 cache 友好性，同时保留 IGA 的几何精度。

---

## 数学框架

### 1. H(curl) 协变 Piola 变换

对 H(curl) 场，物理域和参数域之间的变换为协变 Piola 变换：

$$
\hat{\mathbf{u}}(\xi) = J^T(\xi)\, \mathbf{u}(F(\xi)),
\qquad
\mathbf{u}(x) = J^{-T}(\xi)\, \hat{\mathbf{u}}(\xi)
$$

其中 $J = DF(\xi) \in \mathbb{R}^{3\times3}$ 是几何映射的 Jacobian。

**关键性质**：协变 Piola 变换保持旋度结构：
$$
\widehat{\operatorname{curl}\, \mathbf{u}} = \frac{1}{\det J}\, J\, \operatorname{curl}_\xi \hat{\mathbf{u}}
$$

### 2. 参数域中的 Maxwell 方程（等效变系数问题）

将物理域 Maxwell 弱形式拉回到参数域后得到：

$$
\int_{\hat{\Omega}} \frac{1}{\mu_{\text{eff}}(\xi)}\,
\operatorname{curl}_\xi \hat{\mathbf{u}} \cdot \operatorname{curl}_\xi \hat{\mathbf{v}}\, d\xi
- \omega^2 \int_{\hat{\Omega}} \varepsilon_{\text{eff}}(\xi)\,
\hat{\mathbf{u}} \cdot \hat{\mathbf{v}}\, d\xi = \hat{f}(\hat{\mathbf{v}})
$$

其中等效材料张量为：

$$
\boldsymbol{\mu}_{\text{eff}}(\xi) = \mu\, |\det J|\, J^{-1} J^{-T},
\qquad
\boldsymbol{\varepsilon}_{\text{eff}}(\xi) = \varepsilon\, |\det J|\, J^{-T} J^{-1}
$$

**几何意义**：物理域的曲率在参数域中表现为**各向异性、空间变化的等效介质**。
这正是变换光学（transformation optics）的数学本质——几何映射等价于"空间弯曲"。

### 3. 两种加速策略

#### 策略 A：Yee 网格初始猜测 + IGA 精化（最快实现）

```
Step 1: 在参数域 [0,1]^3 上用常系数 Yee/maxwellb 快速求解
         ε_avg = mean(ε_eff),  μ_avg = mean(μ_eff)
         → 得到参数域场 û_coarse（很快，无矩阵 CUDA kernel）

Step 2: 协变 Piola 逆变换映射到物理域
         u_0(x) = J^{-T}(F^{-1}(x)) û_coarse(F^{-1}(x))

Step 3: 用 u_0 作初始猜测运行 IGA BiCGSTAB
         → 预期迭代次数从 ~20 步降至 ~3-5 步
```

**理论依据**：当几何映射接近仿射（即 J ≈ const）时，常系数 Yee 解经 Piola 变换
后近似满足 IGA 方程，使初始残差 $\|r_0\|/\|b\| \ll 1$，迭代次数大幅减少。

#### 策略 B：参数域变系数矩阵-向量积（更强的预处理）

直接在参数域实现 $\hat{A}\hat{u}$ 的矩阵-向量积：

$$
(\hat{A}\hat{u})_i = \sum_j \hat{A}_{ij} \hat{u}_j
= \int_{\hat{\Omega}} \boldsymbol{\mu}_{\text{eff}}^{-1} \operatorname{curl}_\xi \hat{\phi}_i
  \cdot \operatorname{curl}_\xi \hat{u} \, d\xi
- \omega^2 \int_{\hat{\Omega}} \boldsymbol{\varepsilon}_{\text{eff}} \hat{\phi}_i \cdot \hat{u} \, d\xi
$$

由于参数域是规则格点，可以用类似 maxwellb 的 CUDA kernel 逐点计算，
将每格点的 $J(\xi)$ 预存为常数数组（`gce.Const` 风格）。

整个 SpMV 变为：**无需稀疏矩阵，每次乘法 O(N) 内存访问**，cache 命中率接近 maxwellb。

#### 策略 C：两层 Schwarz（参数域粗空间 + IGA 细空间）

$$
B^{-1} = \underbrace{P_{\text{IGA}}^T A_{\text{IGA}}^{-1} P_{\text{IGA}}}_{\text{IGA 细层（patch LU）}}
+ \underbrace{P_{\text{Yee}}^T A_{\text{Yee}}^{-1} P_{\text{Yee}}}_{\text{Yee 粗层（maxwellb BiCGSTAB）}}
$$

Yee 粗空间捕获低频误差（远场辐射），IGA 细空间处理高频几何误差（曲面附近）。
迁移算子 $P$ 通过 NURBS 插值 + Piola 变换实现。

---

## 实现路线图

### Phase 1：可行性验证（策略 A，2D square-nurbs）

- [ ] 从 IGA 系统提取 NURBS 几何映射 $F$，计算各网格点 $J(\xi)$
- [ ] 在 $[0,1]^2$ 上运行 maxwellb BiCGSTAB（Python/CUDA），得到 $\hat{u}$
- [ ] 实现协变 Piola 逆变换：$u_0 = J^{-T} \hat{u}$，投影为 IGA DOF 系数
- [ ] 对比：以零向量 vs $u_0$ 为初值的 IGA BiCGSTAB 迭代次数

### Phase 2：参数域矩阵-向量积（策略 B，3D cube-nurbs）

- [ ] 预计算 $\boldsymbol{\varepsilon}_{\text{eff}}(\xi_i)$、$\boldsymbol{\mu}_{\text{eff}}(\xi_i)$ 在 Yee 格点上的值
- [ ] 扩展 maxwellb 的 `alpha_step` CUDA kernel 支持逐点张量系数
- [ ] 在 IGA BiCGSTAB 的 `multA` 中替换为此矩阵-向量积（去掉稀疏矩阵）

### Phase 3：两层 Schwarz（策略 C，论文核心贡献）

- [ ] 实现 Yee ↔ IGA 的插值迁移算子
- [ ] 将 Yee-空间 BiCGSTAB 封装为粗层预处理子
- [ ] 理论分析：谱等价性，条件数估计

---

## 论文定位

**标题方向**：
> "Geometry-Aware Reference-Domain Preconditioning for High-Order IGA H(curl)
>  Maxwell Solvers Using Matrix-Free Yee-Grid Acceleration"

**核心贡献**：
1. 建立 IGA H(curl) 与参数域变系数 Yee-Maxwell 的精确等价关系（协变 Piola 框架）
2. 提出三类利用 Yee 网格规则性加速 IGA 迭代的方法（初始猜测、矩阵-向量积、两层 Schwarz）
3. 数值验证：迭代次数减少、GPU 加速比

**与现有工作的区别**：
- AMS/HypreAMS：辅助空间方法，在 PML + 高阶 IGA 下失效
- IGA-RAS：patch LU 代价高，多进程下膨胀
- 本方法：利用 IGA 参数域的规则性，完全绕开显式矩阵组装

---

## 文件结构

```
reference_precon/
├── DESIGN.md              ← 本文档（数学框架）
├── piola_transform.hpp    ← H(curl) 协变 Piola 变换工具函数
├── yee_ref_matvec.py      ← 参数域变系数 Yee MatVec（Python/CUDA 原型）
├── iga_initial_guess.cpp  ← 从 Yee 解构建 IGA 初始猜测
└── test/
    ├── test_piola_2d.cpp  ← 2D square-nurbs 验证
    └── test_piola_3d.cpp  ← 3D cube-nurbs 验证
```
