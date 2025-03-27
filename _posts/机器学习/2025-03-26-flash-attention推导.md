---
categories: [机器学习]
tags: [大模型]     # TAG names should always be lowercase
math: true
---

# Flash Attention 计算正确性数学推导

## 标准注意力计算

给定查询矩阵 $$Q \in \mathbb{R}^{N \times d}$$，键矩阵$$ K \in \mathbb{R}^{N \times d}$$，值矩阵$$V \in \mathbb{R}^{N \times d}$$，标准注意力输出$$O \in \mathbb{R}^{N \times d}$$计算为：

$$
O = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

其中，对每个查询 $ q_i \in Q $，计算注意力权重：

$$
a_i = \text{softmax}\left(\frac{q_i K^T}{\sqrt{d}}\right) = \frac{\exp\left(\frac{q_i K^T}{\sqrt{d}} - m_i\right)}{\sum_{j=1}^N \exp\left(\frac{q_i k_j^T}{\sqrt{d}} - m_i\right)}
$$

这里 $ m_i = \max_j \left(\frac{q_i k_j^T}{\sqrt{d}}\right) $，输出为：

$$
o_i = a_i V = \frac{\sum_{j=1}^N \exp\left(\frac{q_i k_j^T}{\sqrt{d}} - m_i\right) v_j}{\sum_{j=1}^N \exp\left(\frac{q_i k_j^T}{\sqrt{d}} - m_i\right)}
$$

## Flash Attention 分块计算

将 $ K $ 和 $ V $ 分为 $ B $ 块，依次处理每个块 $ K_b, V_b $，维护以下变量：
- $ m^{(b)} $: 前 $ b $ 块的最大值
- $ l^{(b)} $: 前 $ b $ 块的指数和（分母）
- $ o^{(b)} $: 前 $ b $ 块的累积输出

### 初始化

$$
m^{(0)} = -\infty, \quad l^{(0)} = 0, \quad o^{(0)} = 0
$$

### 第 $ b $ 块处理步骤

1. **计算当前块注意力分数**：

   $$
   S_b = \frac{q_i K_b^T}{\sqrt{d}} \quad \in \mathbb{R}^{1 \times n_b}
   $$

2. **更新最大值**：

   $$
   \hat{m}_b = \max\left(m^{(b-1)}, \max(S_b)\right)
   $$

3. **调整前块的指数和与输出**：

   $$
   \text{scale}_\text{prev} = e^{m^{(b-1)} - \hat{m}_b}, \quad \text{scale}_b = e^{\max(S_b) - \hat{m}_b}
   $$

4. **更新分母**：

   $$
   l^{(b)} = \text{scale}_\text{prev} \cdot l^{(b-1)} + e^{\max(S_b) - \hat{m}_b} \cdot \sum_{j=1}^{n_b} e^{S_{bj} - \max(S_b)}
   $$

5. **更新输出**：

   $$
   o^{(b)} = \frac{\text{scale}_\text{prev} \cdot l^{(b-1)} \cdot o^{(b-1)} + \text{scale}_b \cdot \left(e^{S_b - \max(S_b)} V_b\right)}{l^{(b)}}
   $$

6. **更新最大值**：

   $$
   m^{(b)} = \hat{m}_b
   $$

### 最终输出
处理所有块后，得到：

$$
o_i = o^{(B)} = \frac{\sum_{b=1}^B e^{S_b - m^{(B)}} V_b}{\sum_{b=1}^B e^{S_b - m^{(B)}}}
$$

## 数学等价性证明

**目标**：证明分块计算的结果 $ o_i^{(B)} $ 等于标准注意力输出 $ o_i $。

**证明**：

1. **分母一致性**：
    - 分块计算的分母 $ l^{(B)} = \sum_{b=1}^B e^{S_b - m^{(B)}} $，与标准计算的分母一致。

2. **分子一致性**：
    - 分块计算的分子为 $ \sum_{b=1}^B e^{S_b - m^{(B)}} V_b $，与标准计算的分子一致。

3. **递推关系**：
    - 通过归纳法可验证，每次更新后的 $ o^{(b)} $ 均为前 $ b $ 块的加权平均，权重为调整后的指数值。

综上，Flash Attention 的分块计算通过维护最大值、分母和输出的递推关系，保证了与标准注意力计算的数学等价性。