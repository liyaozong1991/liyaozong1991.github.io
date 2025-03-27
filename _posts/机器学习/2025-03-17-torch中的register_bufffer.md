---
categories: 机器学习
math: true
---

在PyTorch中，使用`nn.Parameter`（设置`requires_grad=False`）和`register_buffer`的主要区别如下：

| **特性**                | `nn.Parameter(requires_grad=False)`           | `register_buffer`            |
|-------------------------|-----------------------------------------------|------------------------------|
| **存储位置**            | 属于模型参数 (`model.parameters()`)           | 属于模型缓冲区 (`model.buffers()`)  |
| **梯度计算**            | `requires_grad` 默认是 `True`，需显式关闭     | 没有梯度，不会计算                    |
| **优化器是否处理**      | 会出现在 `model.parameters()` 中，可能被优化器处理（需手动过滤） | 不会出现在 `model.parameters()` 中 |
| **序列化（保存/加载）** | 保存到 `state_dict` 的 `参数` 部分            | 保存到 `state_dict` 的 `缓冲区` 部分  |
| **典型用途**            | 需要固定参数的场景（如迁移学习中的预训练层）  | 持久化非训练张量（如BatchNorm的统计量）     |
| **设备移动**            | 自动跟随模型设备（`model.to(device)`）        | 自动跟随模型设备（`model.to(device)`） |
| **参数数量统计**        | 计入 `model.parameters()` 的总参数量          | 不计入模型参数量                     |

### 关键区别总结：
1. **优化器影响**：
    - `nn.Parameter(requires_grad=False)` 会出现在 `model.parameters()` 中，可能导致优化器误处理（需手动过滤）。
    - `register_buffer` 不会出现在参数列表中，完全绕过优化器。

2. **设计意图**：
    - `nn.Parameter` 本质是模型参数，即使不可训练（如固定预训练权重）。
    - `register_buffer` 用于存储与模型相关但无需训练的状态（如均值和方差）。

3. **序列化区分**：  
   两者均保存在 `state_dict` 中，但参数和缓冲区在语义上是分离的，便于区分用途。

### 代码示例：
```
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 不可训练参数
        self.fixed_param = nn.Parameter(torch.randn(3), requires_grad=False)
        # 注册缓冲区
        self.register_buffer("running_mean", torch.zeros(3))
    
    def forward(self, x):
        # 使用参数和缓冲区
        return x * self.fixed_param + self.running_mean
```