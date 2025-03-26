---
layout: post
comments: true
categories: 机器学习
---

## Self-Attention 与 Cross-Attention 的区别及实现细节

### 应用场景

| 类型            | 应用场景                                                                 | 示例                          |
|-----------------|-------------------------------------------------------------------------|-----------------------------|
| **Self-Attention**  | 单序列内部关系建模                                                      | Transformer 编码器、文本分类       |
| **Cross-Attention** | 两个序列之间的交互建模                                                  | Transformer 解码器、机器翻译/文本生成 |

---

### 核心区别

| 特性               | Self-Attention                          | Cross-Attention                          |
|--------------------|-----------------------------------------|------------------------------------------|
| **输入来源**       | Q/K/V 来自同一输入                      | Q 来自输入 A，K/V 来自输入 B               |
| **序列关系**       | 单序列内部关系                          | 跨序列关系（如编码器-解码器交互）             |
| **典型位置**       | Transformer 编码器                      | Transformer 解码器                        |

---

### Python 实现代码 (PyTorch)

#### 1. Self-Attention 实现

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)  # 拆分为 Q/K/V
        
        # 线性投影 + 多头拆分
        q, k, v = [ 
            t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            for t in qkv
        ]

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        
        # 加权和 + 合并多头
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out(out)
```

#### Cross-Attention 实现

```
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q = nn.Linear(embed_dim, embed_dim)  # Q 来自输入 x
        self.kv = nn.Linear(embed_dim, 2 * embed_dim)  # K/V 来自 encoder_output

    def forward(self, x, encoder_output):
        batch_size, seq_len, _ = x.shape
        
        # 生成 Q/K/V（注意来源不同）
        q = self.q(x)
        k, v = self.kv(encoder_output).chunk(2, dim=-1)

        # 多头拆分
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        
        # 加权和 + 合并多头
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return out
```

### Attention机制实现方式全面对比

| 对比维度           | flash_attention_2                                                                 | flex_attention                                                                   | sdpa                                                                             |
|--------------------|-----------------------------------------------------------------------------------|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| **核心算法**       | 基于分块计算和重计算的IO优化算法                                                  | 动态稀疏注意力机制                                                              | PyTorch内置的scaled_dot_product_attention实现                                   |
| **计算复杂度**     | O(N²)理论复杂度，但通过分块减少实际计算量                                         | 支持稀疏模式时可达O(N√N)                                                        | O(N²)标准实现，但通过硬件加速优化                                               |
| **内存占用**       | 最低（不存储完整attention矩阵）                                                  | 中等（支持稀疏存储）                                                            | 较高（需存储完整attention矩阵）                                                |
| **硬件加速**       | 强CUDA优化                                                                        | 支持多平台适配                                                                  | 深度集成CUDA/cuDNN                                                              |
| **序列长度支持**   | 最优（支持超长序列）                                                              | 中等（依赖稀疏模式）                                                            | 标准长度（受显存限制）                                                          |
| **反向传播支持**   | 需重计算前向结果                                                                  | 原生支持                                                                        | 原生支持                                                                        |
| **扩展性**         | 固定分块策略                                                                      | 支持自定义稀疏模式                                                              | 固定标准实现                                                                    |
| **实现复杂度**     | 高（需手工CUDA优化）                                                              | 中等（需定义稀疏策略）                                                          | 低（直接调用API）                                                               |
| **适用场景**       | 1. 超长序列处理<br>2. 内存敏感场景<br>3. 训练场景                                 | 1. 稀疏注意力需求<br>2. 动态模式切换<br>3. 研究性场景                           | 1. 标准Transformer<br>2. 推理场景<br>3. 快速原型开发                            |
| **精度控制**       | 使用FP16/FP32混合精度                                                             | 原生支持自动混合精度                                                            | 依赖框架自动混合精度                                                            |
| **框架依赖**       | 需要定制CUDA扩展                                                                  | 需要特定框架支持                                                                | 深度集成PyTorch                                                                 |
| **典型应用案例**   | 1. LLM训练<br>2. 长文本处理                                                       | 1. 视觉Transformer<br>2. 图神经网络                                           | 1. 标准BERT/GPT<br>2. 移动端部署                                               |

#### 关键结论：

1. **训练场景优先**：`flash_attention_2`在内存效率和长序列处理上表现最优
2. **动态稀疏需求**：`flex_attention`提供最灵活的注意力模式配置
3. **快速开发推荐**：`sdpa`凭借PyTorch深度集成实现最佳开发效率
4. **硬件适配性**：`sdpa > flash_attention_2 > flex_attention`
5. **内存敏感场景**：`flash_attention_2`的内存优化策略可节省30-50%显存

> 注：实际性能表现需结合具体硬件配置（如A100/H100对flash_attention_2有特殊优化）和任务特性