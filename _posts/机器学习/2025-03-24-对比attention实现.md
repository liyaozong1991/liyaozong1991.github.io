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