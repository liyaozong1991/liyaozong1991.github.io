---
categories: [机器学习]
tags: [推荐系统, Scaling Laws, 特征交叉, 工业实践, 快手]
math: true
title: "UniMixer: 统一推荐系统三大 Scaling 架构的理论框架与实践"
---

**论文**: UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems  
**链接**: [https://arxiv.org/abs/2604.00590](https://arxiv.org/abs/2604.00590)  
**机构**: 快手 (Kuaishou Technology)  
**时间**: 2026年4月

## 1. 问题背景

LLM 中 Scaling Laws 的成功激发了推荐系统社区探索类似的扩展范式——通过堆叠更多层、增加参数量来提升排序模型性能。然而，推荐系统与 NLP 存在一个根本性差异：**特征空间的异构性（Heterogeneity）**。LLM 中所有 token 共享统一的 embedding 空间，而推荐系统的输入包含了类别特征、数值特征、ID 特征、交叉特征、序列特征等来自完全不同语义空间的异构特征。这意味着直接将 Transformer 搬到推荐场景是不可行的——在两个异构语义空间之间计算内积相似度缺乏合理的物理意义。

围绕这一挑战，当前工业界形成了三大 Scaling 架构范式：

| 范式 | 代表工作 | 核心思路 | 局限性 |
| --- | --- | --- | --- |
| **Attention-Based** | HiFormer, FAT, HHFT | 为每个 token 构建独立的 Q/K/V 投影，实现 token 级异构特征交互 | 注意力权重尖锐稀疏，梯度传播困难；二次复杂度 |
| **TokenMixer-Based** | RankMixer, TokenMixer-Large | 用无参数、基于规则的 token 混合操作替代注意力 | 缺乏可学习性和场景适应性；要求 $T=H$ |
| **FM-Based** | Wukong, Kunlun | 通过 FM 模块计算特征交互，结合线性投影扩展 | 受限于显式低阶交互，大规模 Scaling 时性能提升有限 |

这三类方法建立在完全不同的计算模块之上，但都展现出了 Scaling 能力。这自然引出一个核心问题：**能否构建一个统一的 Scaling 模块，融合三类方法的优势，并建立统一的理论框架？**

UniMixer 正是对这个问题的回答。其核心贡献包括：(1) 通过参数化 TokenMixer 的排列矩阵，揭示了 TokenMixer 的本质特征交互模式；(2) 建立统一的理论框架，桥接三类方法的差异与联系；(3) 设计轻量化的 UniMixing-Lite 模块，同时利用 Attention 和 TokenMixer 的优势实现更高的 Scaling 效率。

---

## 2. 核心方法

### 2.1 Feature Tokenization

输入特征首先按语义类别划分为 $N$ 个不相交的特征域（Feature Domain）：

$$
\textbf{X} = \Big[\underset{\text{User Profile}}{\underbrace{\textbf{x}_{U}^{(1)}, \dotsc, \textbf{x}_{U}^{(n_U)}}}, \underset{\text{Item Features}}{\underbrace{\textbf{x}_{I}^{(1)}, \dotsc, \textbf{x}_{I}^{(n_I)}}}, \underset{\text{Behavior Sequence}}{\underbrace{\textbf{x}_{B}^{(1)}, \dotsc, \textbf{x}_{B}^{(n_B)}}}, \dotsc \Big]
$$


每个特征域通过 Embedding 层映射到不同维度的 embedding 向量 $\textbf{e}\_n = \text{Embedding}(\textbf{X}\_{\text{domain}}) \in \mathbb{R}^{d\_{\text{domain}}}$，然后所有域的 embedding 拼接为一个向量 $\textbf{E}$。关键步骤是将 $\textbf{E}$ 均匀切分成若干 block，每个 block 通过 **token 级独立线性层** 投影到统一维度 $D$：


$$
\boldsymbol{x}_i = W^{\text{proj}}_i \textbf{E}_{di:di+d} + \textbf{b}^{\text{proj}}_i \in \mathbb{R}^D
$$

这里 $W^{\text{proj}}\_i$ 是 token 级独立参数（不同 token 用不同的投影矩阵），目的是保留异构特征的域特异性语义，同时对齐到统一维度。最终得到输入隐状态 $X \in \mathbb{R}^{T \times D}$，其中 $T$ 为 token 数量，$D$ 为 token 维度。

### 2.2 TokenMixer 的等价参数化：核心发现

UniMixer 最关键的理论贡献是发现了 **TokenMixer 操作可以等价表示为一个排列矩阵与展平输入的乘积**。

回顾 TokenMixer 的操作过程：对输入 $X = [\boldsymbol{x}\_1; \dotsc; \boldsymbol{x}\_T] \in \mathbb{R}^{T \times D}$，先将每个 token $\boldsymbol{x}\_t$ 沿维度方向均匀切分为 $H$ 个 head：

$$
[\boldsymbol{x}_t^{(1)} | \boldsymbol{x}_t^{(2)} | \dotsc | \boldsymbol{x}_t^{(H)}] = \text{SplitHead}(\boldsymbol{x}_t)
$$

然后跨 token 重新组合——将所有 token 的第 $h$ 个 head 拼接成新的 token：

$$
\boldsymbol{s}^h = \text{concat}(\boldsymbol{x}_1^{(h)}, \boldsymbol{x}_2^{(h)}, \dotsc, \boldsymbol{x}_T^{(h)}) \in \mathbb{R}^{TD/H}
$$

其中 $H$ 必须等于 $T$。

论文的关键发现是：这个看似简单的重排操作，**本质上等价于一个大排列矩阵 $W^{\text{perm}} \in \mathbb{R}^{TD \times TD}$ 与展平输入向量 $\text{flatten}(X)$ 的矩阵乘法**：

$$
\text{TokenMixer}(X) = \text{reshape}(W^{\text{perm}} \cdot \text{flatten}(X))
$$

论文在附录 A 中给出了一个 $T=2, D=6$ 的具体数值示例来直观说明这一点。当 $X = \begin{bmatrix} x\_1 & x\_2 & x\_3 & x\_4 & x\_5 & x\_6 \\\\ x\_7 & x\_8 & x\_9 & x\_{10} & x\_{11} & x\_{12} \end{bmatrix}$ 时，TokenMixer 的输出为 $\begin{bmatrix} x\_1 & x\_2 & x\_3 & x\_7 & x\_8 & x\_9 \\\\ x\_4 & x\_5 & x\_6 & x\_{10} & x\_{11} & x\_{12} \end{bmatrix}$，这个变换可以精确用一个 $12 \times 12$ 的排列矩阵来表示。

更重要的是，论文发现了这个排列矩阵 $W^{\text{perm}}$ 的四个关键数学性质：

**1. 可压缩性（Compressibility）**。$W^{\text{perm}}$ 可以等价分解为两个更小矩阵的 Kronecker 积：$W^{\text{perm}} = G \otimes I$，其中 $G \in \mathbb{R}^{T^2 \times T^2}$ 控制全局 token 间混合模式，$I \in \mathbb{R}^{D/T \times D/T}$ 是单位矩阵（对应局部维度内不做变换）。这将参数规模从 $O(T^2 D^2)$ 降低到 $O(T^4 + (D/T)^2)$，其中 $T$ 通常远小于 $D$。

**2. 双随机性（Doubly Stochasticity）**。每行每列的元素之和均为 1：$\sum\_p w^{\text{perm}}\_{pq} = 1, \sum\_q w^{\text{perm}}\_{pq} = 1$。这意味着排列矩阵保持信号的总能量守恒，不会放大或缩小输入信号。

**3. 稀疏性（Sparsity）**。每行/每列恰好只有一个非零元素（值为 1）。这意味着每个输出位置恰好对应一个输入位置，是一种严格的一对一映射。

**4. 对称性（Symmetry）**。当 $T = H$ 时，$W^{\text{perm}} = W^{\text{perm}\mathsf{T}}$，即排列矩阵是对称的——操作执行两次等于回到原始状态（对合排列）。当 $T \neq H$ 时，矩阵不对称。

这四个性质的发现，为从 TokenMixer 出发构建统一框架提供了理论基础。一个自然的想法是：如果将排列矩阵 $W^{\text{perm}}$ 替换为可学习的参数矩阵，同时尽量保持这些良好性质，就能将 TokenMixer 从"基于规则"升级为"可学习"。

### 2.3 UniMixing 模块

直接参数化 $W^{\text{perm}}$ 面临三个挑战：(1) Kronecker 积重建完整矩阵仍然会产生 $[TD, TD]$ 的中间变量，GPU 内存消耗巨大；(2) 如何保证学到的参数满足双随机性、稀疏性和对称性；(3) 如何设计统一模块融合三类方法的优势。

UniMixing 的设计从重新定义 block 结构开始。不再拘泥于 TokenMixer 中 $T$ 和 $D$ 的划分，而是定义一个 **block 大小 $B$** 和 **block 数量 $L // B$**（$L$ 为输入 embedding 总维度）。模块包含两组可学习参数：

- **全局混合矩阵 $W\_G \in \mathbb{R}^{(L//B) \times (L//B)}$**：控制 block 间的交互模式
- **局部混合矩阵 $\\{W\_B^i\\}\_{i=1}^{L//B}$，每个 $W\_B^i \in \mathbb{R}^{B \times B}$**：控制每个 block 内部的特征交互模式

与原始 TokenMixer 中所有 block 共享同一个单位矩阵 $I$ 不同，UniMixing 为**每行分配独立的参数矩阵 $W\_B^i$**，使不同 block 拥有不同的局部交互模式。完整表达式为：

$$
\text{UniMixing}(X) = \text{reshape}\Big(\Big(W_G \otimes \{W_B^i\}_{i=1}^{L//B}\Big) \text{flatten}(X), 1, L\Big)
$$

其中 $\otimes$ 是广义 Kronecker 积。

#### 计算管线优化

直接计算上式仍然会产生 $[L, L]$ 的中间矩阵。论文通过代数恒等变换，将计算拆解为两步（论文附录 B 给出了完整证明）：

**第一步（局部交互）**：将展平的 embedding 向量均匀切分为 $L//B$ 个大小为 $B$ 的向量 $\\{\boldsymbol{x}\_i\\}$，分别与对应的局部混合矩阵相乘：

$$
H = \begin{bmatrix} \boldsymbol{x}_1 W_B^1 \\ \vdots \\ \boldsymbol{x}_{L/B} W_B^{L/B} \end{bmatrix} \in \mathbb{R}^{(L/B) \times B}
$$

**第二步（全局交互）**：将全局混合矩阵 $W\_G$ 作用于局部交互结果：

$$
\text{UniMixing}(X) = \text{reshape}(W_G \cdot H, 1, L)
$$

这个优化将计算复杂度从 $O(L^2)$ 降低到 $O(L^2/B + LB)$，避免了创建大尺寸中间变量。直觉理解：$W\_B^i$ 负责 block 内部的特征变换（类似于 Attention 中的 Value 投影），$W\_G$ 负责 block 间的信息交换（类似于注意力权重矩阵）。

#### 约束条件的实现

为了保持排列矩阵的良好性质，论文对学到的参数施加三重约束：

**对称性约束**。通过矩阵对称化实现：

$$
\tilde{W}_G = \frac{W_G + W_G^{\mathsf{T}}}{2}, \quad \tilde{W}_B^i = \frac{W_B^i + W_B^{i\mathsf{T}}}{2}
$$

**双随机性约束**。通过 Sinkhorn-Knopp 迭代实现——先用指数函数确保所有元素为正，然后交替对行和列做归一化，使每行每列之和均为 1：

$$
\bar{W}_G = \text{Sinkhorn-Knopp}\Big(\frac{\tilde{W}_G}{\tau}\Big), \quad \bar{W}_B^i = \text{Sinkhorn-Knopp}\Big(\frac{\tilde{W}_B^i}{\tau}\Big)
$$

**稀疏性约束**。通过温度系数 $\tau$ 控制——$\tau$ 越小，Sinkhorn-Knopp 输出越接近独热分布（越稀疏），$\tau$ 越大，分布越均匀。论文消融实验证明低温度（高稀疏性）对模型性能有显著正面效果。

最后加上残差连接和 RMSNorm：

$$
O = \text{RMSNorm}(X + \text{UniMixing}(X))
$$

### 2.4 统一理论框架

UniMixer 最重要的理论贡献是将三类 Scaling 模块统一到同一个框架下。关键观察是：

**UniMixing 与 Heterogeneous Attention 的联系**。对比 Attention 中的 Value 投影 $V\_h$ 和 UniMixing 中的局部交互矩阵 $H$：如果 block 数量 $L//B$ 设为 $T$，且 $W\_V^{ih}$ 与 $W\_B^i$ 维度相同，则 **UniMixing 的局部交互投影等价于异构注意力的 Value 投影**（在 $W\_V^i = W\_B^i$ 条件下）。同时，$W\_G$ 的维度和角色与注意力权重矩阵一致，只是附加了双随机性、稀疏性和对称性约束。

**Attention 与 FM 的联系**。Wukong 的 FM 组件 $\text{FM}(X) = XX^{\mathsf{T}}Y$ 可以改写为 $XI(XI)^{\mathsf{T}}Y$。当注意力机制中 $W\_Q = I$、$W\_K = I$、且 Value 不依赖于输入 $X$（即 $V\_h = Y$ 为固定参数矩阵）时，**Attention 退化为 FM 模块**。

这些联系允许在以下统一框架下表达所有方法：

$$
\text{UniMixing}(X) = \text{reshape}\Bigg(\underset{\text{Global Mixing Pattern}}{\underbrace{G(X, W_G)}} \underset{\text{Local Mixing Pattern}}{\underbrace{\begin{bmatrix} \boldsymbol{x}_1 W_B^1 \\ \vdots \\ \boldsymbol{x}_{L/B} W_B^{L/B} \end{bmatrix}}}, 1, L\Bigg)
$$

各方法在这一框架下的差异总结如下：

| 方法 | Local Mixing Pattern | Global Mixing Pattern $G(X, W\_G)$ |
| --- | --- | --- |
| Self-Attention | $XW\_V$ | $\text{softmax}(\frac{(XW\_Q)(XW\_K)^{\mathsf{T}}}{\sqrt{d}})$ |
| Heterogeneous Attention | $X\tilde{W}\_V$ | $\text{softmax}(\frac{(X\tilde{W}\_Q)(X\tilde{W}\_K)^{\mathsf{T}}}{\sqrt{d}})$ |
| TokenMixer | $X$ （无投影） | $G$ （固定矩阵，不依赖输入） |
| FM | $Y$ （固定参数，不依赖输入） | $XI(XI)^{\mathsf{T}}$ |

从这个框架可以清晰看出：(1) Attention 和 FM 的全局混合模式都依赖于输入 $X$（数据驱动），而 TokenMixer 的全局混合模式是固定的；(2) Attention 的局部混合模式是输入相关的（通过 Value 投影），而 TokenMixer 直接使用原始输入、FM 使用固定参数。UniMixing 的设计选择是：**局部混合学习异构投影（类似 Attention），全局混合学习可优化但输入无关的混合模式（类似 TokenMixer），同时附加双随机性等约束保证稳定性**。

### 2.5 UniMixing-Lite：轻量化设计

当 block 粒度变细时，局部交互矩阵 $W\_B^i$ 的数量增加，全局交互矩阵 $W\_G$ 变大，导致局部交互模式冗余、参数效率下降。UniMixing-Lite 通过两个策略解决这一问题：

**1. 基底合成的局部混合矩阵（Basis-Composed Local Mixing）**。定义一组基底矩阵 $\\{Z\_\ell\\}\_{\ell=1}^b$ 和 block 级权重向量 $\\{\boldsymbol{\omega}^i\\}\_{i=1}^{L//B}$，每个 block 的局部混合矩阵通过基底的线性组合动态生成：

$$
W_B^{*i} = \text{Sinkhorn-Knopp}\Big(\sum_{\ell=1}^b \omega_\ell^i Z_\ell\Big)
$$

其中 $b$ 为基底数量，$\boldsymbol{\omega}^i = [\omega\_1^i, \dotsc, \omega\_b^i]$ 为 block 级系数。这样只需存储 $b$ 个基底矩阵和 $(L//B)$ 个权重向量，而非 $(L//B)$ 个完整矩阵。不同 block 通过不同的组合系数 $\boldsymbol{\omega}^i$ 获得差异化的局部交互模式，同时共享基底降低了参数冗余。

**2. 低秩近似的全局混合矩阵**。将 $W\_G$ 分解为两个低秩矩阵的乘积：

$$
W_r = \text{Sinkhorn-Knopp}(A_G B_G), \quad A_G \in \mathbb{R}^{(L//B) \times r}, B_G \in \mathbb{R}^{r \times (L//B)}
$$

其中 $r$ 为秩。论文中的可视化实验（Fig. 5）显示，虽然使用了低秩近似和基底合成，但 Sinkhorn-Knopp 操作仍能确保重建的矩阵接近满秩——这是因为 Sinkhorn-Knopp 的迭代归一化过程天然具有"分散"元素值的效果。

UniMixing-Lite 的完整表达式：

$$
\text{UniMixing-Lite}(X) = \text{reshape}\Big(W_r \begin{bmatrix} \boldsymbol{x}_1 W_B^{*1} \\ \vdots \\ \boldsymbol{x}_{L/B} W_B^{*L/B} \end{bmatrix}, 1, L\Big)
$$

核心设计理念是：**保留 TokenMixer 低参数化全局交互的优势，同时融入 Attention 对异构特征的局部交互能力**。

### 2.6 SiameseNorm：深层网络训练稳定性

现有的 RankMixer 架构在增加深度时 Scaling 效果有限——RankMixer-4-Blocks 的性能甚至**低于** RankMixer-2-Blocks（Table 4 中 AUC 下降 0.1066%）。TokenMixer-Large 虽然尝试通过间隔残差和辅助 loss 缓解这个问题，但没有从根本上解决。

UniMixer 引入 SiameseNorm 来解决 Pre-Norm 和 Post-Norm 之间的矛盾。SiameseNorm 维护两个耦合流 $\bar{X}\_\ell$ 和 $\bar{Y}\_\ell$（均初始化为输入 embedding），在每一层执行：

$$
\tilde{Y}_\ell = \text{RMSNorm}(\bar{Y}_\ell), \quad O_\ell = \text{UniMixer}(\bar{X}_\ell + \tilde{Y}_\ell)
$$

$$
\bar{X}_{\ell+1} = \text{RMSNorm}(\bar{X}_\ell + O_\ell), \quad \bar{Y}_{\ell+1} = \bar{Y}_\ell + O_\ell
$$

最终融合：$X\_{\text{output}} = \bar{X}\_M + \text{RMSNorm}(\bar{Y}\_M)$。

直觉理解：$\bar{X}$ 流在每层做 RMSNorm（类似 Pre-Norm，梯度传播稳定），$\bar{Y}$ 流做纯残差累加（类似 Post-Norm，信号强度不衰减）。两个流互相馈入、最终融合，兼顾了训练稳定性和信号保持。实验证明 SiameseNorm 对 4-blocks 和 8-blocks 的深层配置至关重要——使用 SiameseNorm 后，UniMixer-Lite-4-Blocks 相比 2-Blocks 获得 +0.1575% AUC 提升，而 RankMixer-4-Blocks 则下降了 -0.1066%。

### 2.7 Pertoken SwiGLU

UniMixing 模块之后，采用 token 级独立的 SwiGLU 来建模不同 token 间的特征异构性：

$$
\text{pSwiGLU}(\boldsymbol{o}_i) = W_{\text{down}}^i \Big((W_{\text{up}}^i \boldsymbol{o}_i + b_{\text{up}}^i) \odot \text{Swish}(W_{\text{gate}}^i \boldsymbol{o}_i + b_{\text{gate}}^i)\Big) + b_{\text{down}}^i
$$

其中 $W\_{\text{up}}^i, W\_{\text{gate}}^i \in \mathbb{R}^{B \times nB}$，$W\_{\text{down}}^i \in \mathbb{R}^{nB \times B}$，$n$ 为扩展倍数。注意每个 token 拥有独立的权重矩阵（上标 $i$ 表示 token 索引），这与标准 Transformer FFN 中所有 token 共享权重的设计不同——目的是保留推荐场景下不同特征域的语义差异。

### 2.8 训练策略：温度退火与模型预热

温度系数 $\tau$ 的选择面临两难：低温度产生更稀疏的权重（对性能有利），但也导致梯度稀疏、优化困难。论文提出两种策略：

**线性温度退火**。从较高初始温度 $\tau\_{\text{start}} = 1.0$ 线性降到 $\tau\_{\text{end}} = 0.05$：

$$
\tau_j = \max\Big\{\tau_{\text{start}} - \frac{(\tau_{\text{start}} - \tau_{\text{end}}) j}{J}, \tau_{\text{end}}\Big\}
$$

**两阶段预热策略**。当数据量不足时，线性退火可能导致早期高温探索不充分或晚期低温优化次优。替代方案是：先用高温 $\tau = 1.0$ 完整训练模型（cold-start），再用低温 $\tau = 0.05$ 以高温模型的权重作为初始化进行重训练。消融实验（Table 3）显示，去掉模型预热导致 AUC 下降 0.0856%，仅次于去掉温度系数本身的影响（-0.1645%），说明预热策略对训练质量至关重要。

---

## 3. 实验

### 3.1 实验设置

- **数据集**：快手广告投放场景的真实训练数据，包含超过 7 亿用户样本、收集周期一年，涵盖数百种异构特征。标签为用户次日留存（是否在首次激活次日回到快手 App）。
- **评估指标**：AUC、UAUC（User-Level AUC）衡量模型性能；参数量、FLOPs/Batch、MFU 衡量效率。
- **训练配置**：40 GPU 混合分布式训练，所有模型使用一致的 Adam 优化器（学习率 0.001）。
- **基线**：Heterogeneous Attention、HiFormer、Wukong、FAT、RankMixer、TokenMixer-Large。

### 3.2 主实验结果（~100M 参数级别）

| 模型 | AUC | $\Delta$AUC | UAUC | $\Delta$UAUC | 参数量 | FLOPs/Batch |
| --- | --- | --- | --- | --- | --- | --- |
| Heterogeneous Attention（基线） | 0.744577 | – | 0.733829 | – | 132.7M | 1.68T |
| HiFormer | 0.741685 | -0.2892% | 0.731086 | -0.2743% | 107.5M | 1.37T |
| Wukong | 0.744477 | -0.0100% | 0.733849 | +0.0020% | 107.1M | 1.40T |
| FAT | 0.744883 | +0.0306% | 0.734280 | +0.0451% | 138.4M | 1.83T |
| RankMixer | 0.749329 | +0.4752% | 0.738938 | +0.5109% | 135.5M | 1.68T |
| TokenMixer-Large | 0.748410 | +0.3833% | 0.737940 | +0.4111% | 103.3M | 1.27T |
| **UniMixer-2B 67.5M** | **0.749770** | **+0.5193%** | **0.739331** | **+0.5502%** | **67.5M** | 2.07T |
| **UniMixer-2B 101.5M** | **0.750238** | **+0.5661%** | **0.739983** | **+0.6154%** | 101.5M | 2.50T |
| **UniMixer-Lite-2B 42.4M** | **0.751121** | **+0.6544%** | **0.740739** | **+0.6910%** | **42.4M** | 2.17T |
| **UniMixer-Lite-4B 38.2M** | **0.752327** | **+0.7750%** | **0.742091** | **+0.8190%** | **38.2M** | 1.26T |
| **UniMixer-Lite-4B 84.5M** | **0.752718** | **+0.8141%** | **0.742530** | **+0.8701%** | 84.5M | 4.24T |

几个关键发现：

**1. 参数效率极高**。UniMixer-Lite-2B 仅用 42.4M 参数就超过了所有 ~100M 参数的 SOTA 模型（包括 135.5M 的 RankMixer）。UniMixer-Lite-4B 38.2M 更是以 **28.6%** 的参数量（相对 RankMixer）取得了 +0.3% 的 AUC 提升和更低的 FLOPs（1.26T vs 1.68T）。

**2. 深度 Scaling 有效**。UniMixer-Lite-4-Blocks 比 2-Blocks 有持续提升，而 RankMixer-4-Blocks 反而退化（Table 4），验证了 SiameseNorm 对深度 Scaling 的关键作用。

**3. FLOPs 并非瓶颈**。UniMixer 系列的 FLOPs 偏高（因为可学习的局部混合矩阵引入了额外计算），但由于参数量大幅减少，整体 Scaling 效率（参数效率 + 计算效率综合考量）更优。

### 3.3 Scaling Laws 拟合

论文选取最强 SOTA（RankMixer）与 UniMixer/UniMixer-Lite 进行 Scaling Laws 对比，拟合的幂律关系为：

| 架构 | 参数 Scaling 指数 | FLOPs Scaling 指数 |
| --- | --- | --- |
| RankMixer | 0.116043 | 0.116635 |
| UniMixer | 0.131973 | 0.125702 |
| **UniMixer-Lite** | **0.141903** | **0.135327** |

**Scaling 指数是衡量 Scaling 效率的核心指标**——它决定了每增加一个数量级的参数/FLOPs 能带来多大的性能提升。UniMixer-Lite 的参数 Scaling 指数比 RankMixer 高出 **22.3%**（0.141903 vs 0.116043），意味着在参数量增大的过程中，UniMixer-Lite 的 AUC 增长速度显著更快。

从幂律系数来看，UniMixer-Lite 在系数和指数上都是最优的，说明它不仅增长更快（指数更大），而且在相同参数量下的绝对性能也更高（系数更大）。

### 3.4 消融实验

Table 3 展示了 6.57M 参数量下各组件的消融结果（基于 UniMixer，非 Lite）：

| 配置 | $\Delta$AUC | $\Delta$UAUC |
| --- | --- | --- |
| 完整 UniMixer | – | – |
| 去掉温度系数 | **-0.1645%** | -0.1490% |
| 去掉模型预热 | -0.0856% | -0.0837% |
| 去掉对称性约束 | -0.0573% | -0.0570% |
| 去掉 block 级独立局部权重 | -0.0436% | -0.0240% |
| SiameseNorm → Post Norm | -0.0273% | -0.0357% |

**温度系数的影响最大**（-0.1645%），说明稀疏性对模型性能至关重要——这与直觉一致：稀疏的混合模式意味着更清晰的特征交互路径，减少了噪声信号的干扰。模型预热排第二（-0.0856%），说明训练策略对于最终性能也有显著影响。对称性约束的贡献（-0.0573%）验证了排列矩阵对合性质的重要性。

### 3.5 UniMixing-Lite 的超参数分析

**基底数量 $b$ 的影响**：从 $b=2$ 到 $b=4$，AUC 提升 +0.1002%；从 $b=4$ 到 $b=8$，仅提升 +0.0053%。说明 4 个基底矩阵已经足以覆盖主要的局部交互模式。同时参数增量极小（4.968M → 4.98M），体现了基底合成的参数效率。

**秩 $r$ 的影响**：从 $r=2$ 到 $r=256$，AUC 单调提升（+0.0971%），但增速低于基底数量的影响。这说明在 UniMixing-Lite 中，**增加局部交互多样性（增大 $b$）比增加全局交互容量（增大 $r$）更有效**——推测原因是推荐场景中异构特征的域间差异主要需要通过差异化的局部投影来捕获。

**深度 Scaling**：UniMixer-Lite-8-Blocks 相比 4-Blocks 仅有微小提升（+0.0072% AUC），表明在 4-Blocks 时深度维度的 Scaling 已趋于饱和。论文进一步指出，**沿深度 Scaling 比沿宽度 Scaling 更高效**——4-Blocks 配置在相同参数预算下比 2-Blocks 的宽配置效果更好。

### 3.6 全局与局部混合矩阵可视化

论文可视化了 UniMixer-Lite 的重建矩阵（input embedding 维度 768，block size 为 6，$W\_G \in \mathbb{R}^{128 \times 128}$，$W\_B^i \in \mathbb{R}^{6 \times 6}$，$A\_G \in \mathbb{R}^{128 \times 16}$，$B\_G \in \mathbb{R}^{16 \times 128}$）：

- **$\tau = 1.0$ 时**：全局和局部矩阵的分布较为均匀、平滑，意味着各 block 间和 block 内的交互权重差异不大；
- **$\tau = 0.05$ 时**：矩阵呈现显著的稀疏、尖锐分布，交互集中在特定的 block 对和维度对上。

低温度下更稀疏的交互分布对应更优的模型性能，说明推荐场景中的有效特征交互是稀疏的——并非所有特征域之间都需要交互，模型需要学会"选择性地"关注关键交互对。这与注意力机制中观察到的现象类似，但 UniMixer 通过温度控制 + Sinkhorn-Knopp 约束提供了一种更可控的稀疏化方式。

### 3.7 线上 A/B 测试

UniMixer 和 UniMixer-Lite 已部署在快手多个广告投放场景。线上指标使用 30 天观测窗口内的累计活跃天数（CAD，D1-D30，排除安装当天）。结果：**多场景平均 CAD 提升超过 15%**。

---

## 4. 总结与思考

UniMixer 最核心的贡献是**将 TokenMixer 的排列操作等价转化为矩阵形式，进而发现三类 Scaling 模块可以统一到 "全局混合模式 × 局部混合模式" 的框架下**。这不仅仅是一个抽象的数学发现——它直接指导了 UniMixing 模块的设计：通过可学习的参数替代固定排列矩阵，同时保持双随机性、稀疏性、对称性等良好性质，实现了 TokenMixer 的"可学习化升级"。

**排列矩阵的 Kronecker 积分解是一个精巧的数学洞察**。原始的 $TD \times TD$ 排列矩阵直接参数化显然不可行（$O(T^2D^2)$ 的参数量和计算量），但 Kronecker 积分解将其分离为"全局 block 间交互"和"局部 block 内交互"两个独立维度，参数量降到 $O(T^4 + (D/T)^2)$。更重要的是，这种分离揭示了一个结构性洞察：推荐系统中的特征交互天然具有"层次化"结构——block 间交互对应不同语义域之间的信息交换，block 内交互对应同一语义域内部的特征变换。

**Sinkhorn-Knopp 约束的使用值得深入思考**。双随机矩阵约束确保了混合操作的非扩张性（不放大信号），这对深层网络的训练稳定性至关重要——与 DeepSeek-V4 中 mHC 对 $B\_l$ 矩阵施加双随机约束的思路不谋而合。但一个有趣的开放问题是：Sinkhorn-Knopp 约束是否过于严格？双随机矩阵要求每行每列之和严格为 1，这可能限制了模型表达某些非对称交互模式的能力（例如，某些特征域天然是"信息源"而非"信息汇"）。放松为仅要求列和为 1（右随机矩阵）或引入可学习的缩放因子，可能带来进一步的改进。

**UniMixer-Lite 的基底合成策略值得关注**。通过 $b$ 个共享基底的线性组合来生成 block 级局部混合矩阵，本质上是在参数效率和交互多样性之间做权衡。实验显示 $b=4$ 基本饱和，说明推荐场景中局部交互模式的多样性是有限的——这与直觉一致：无论特征域有多少，底层的交互模式（缩放、旋转、选择性屏蔽等）是有限的几种。这个设计也可以看作是对注意力机制中多头设计的一种"参数化"替代——多头注意力通过不同的 Q/K/V 投影提供多样性，而基底合成通过不同的组合系数提供多样性。

**深度 Scaling 的成功直接得益于 SiameseNorm 的引入**。RankMixer-4-Blocks 性能退化而 UniMixer-Lite-4-Blocks 持续提升的对比实验是最有力的证据。SiameseNorm 的双流设计（Pre-Norm 流保稳定、Post-Norm 流保信号）是一种优雅的折中方案。但论文没有讨论 8-Blocks 时深度 Scaling 接近饱和的原因——推测这可能与推荐任务本身的复杂度上限有关：不同于语言建模中几乎无限的组合复杂度，CTR 预测中有效的特征交互阶数可能是有限的（例如三阶或四阶交互已经足够），更深的网络只是在拟合噪声而非学到新的交互模式。

**一个值得指出的局限性是计算成本**。虽然 UniMixer 在参数效率上显著优于基线，但 FLOPs 并不低——UniMixer-Lite-4B 84.5M 的 FLOPs 为 4.24T，远高于 RankMixer 的 1.68T 和 TokenMixer-Large 的 1.27T。在工业场景中，推理延迟和计算成本往往是硬约束，参数量小但 FLOPs 大的模型在部署时可能并不占优。论文的 MFU（Model FLOPs Utilization）指标没有在主表中报告具体数值，这一点值得进一步关注。不过从另一个角度看，UniMixer-Lite-4B 38.2M 配置实现了 1.26T FLOPs（低于 RankMixer 的 1.68T）同时 AUC 提升 +0.3%，说明在合理的配置选择下，计算成本可以控制在可接受范围内。

**论文提出的统一框架为推荐系统 Scaling 研究提供了一个有价值的理论视角**，但这个框架目前是静态的——仅分析了各方法在单层内的等价关系，没有涉及跨层的交互模式演化。一个自然的延伸方向是：不同层是否应该使用不同类型的 Scaling 模块（类似于 DeepSeek-V4 中 CSA 和 HCA 的交替使用）？例如，浅层使用 Attention 风格的数据驱动全局混合（捕获输入相关的交互），深层使用 TokenMixer 风格的固定全局混合（提取更抽象的组合模式），这种跨层异构设计可能带来更好的 Scaling 效率。
