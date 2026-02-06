---
categories: [机器学习]
tags: [推荐系统, 大模型]
math: true
title: 推荐系统中item的token化方案演进
---

随着大语言模型（LLM）在各领域的广泛应用，如何将推荐系统中的item有效地转化为token序列，成为了生成式推荐系统的核心问题。本文梳理了近年来推荐系统中item token化方案的演进历程，详细介绍了各类代表性论文的核心思想。

## 1. 背景与挑战

传统推荐系统使用随机分配的数字ID来表示item，这种方式存在以下问题：
- **语义缺失**：随机ID无法捕捉item之间的语义相似性
- **泛化能力差**：对于冷启动item难以有效表示
- **与LLM不兼容**：随机ID对预训练语言模型来说是无意义的token

因此，如何设计有效的item token化方案，成为了生成式推荐系统的关键挑战。核心矛盾在于：
- **记忆能力（Memorization）**：需要区分特定item
- **泛化能力（Generalization）**：需要捕捉语义相似性

---

## 2. 开创性工作（2022年）

### 2.1 DSI: Differentiable Search Index

**论文**: Transformer Memory as a Differentiable Search Index  
**链接**: [https://arxiv.org/abs/2202.06991](https://arxiv.org/abs/2202.06991)  
**发表**: NeurIPS 2022

**核心思想**:
- 提出将信息检索任务建模为序列到序列（seq2seq）问题
- 使用单个Transformer模型将query直接映射到document ID
- 探索了多种document ID表示方式：原子ID、朴素字符串、语义结构化ID等
- 相比双塔模型，Hits@1提升超过20个百分点

这是生成式检索领域的开创性工作，为后续推荐系统的token化方案提供了重要启发。

### 2.2 NCI: Neural Corpus Indexer

**论文**: A Neural Corpus Indexer for Document Retrieval  
**链接**: [https://arxiv.org/abs/2206.02743](https://arxiv.org/abs/2206.02743)  
**发表**: NeurIPS 2022 (Outstanding Paper)

**核心思想**:
- 端到端的神经网络直接生成相关文档ID
- 引入前缀感知的权重自适应解码器
- 使用语义文档标识符（Semantic Document Identifiers）
- 在NQ320k数据集上Recall@1提升21.4%

### 2.3 P5: Pretrain, Personalized Prompt & Predict

**论文**: Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5)  
**链接**: [https://arxiv.org/abs/2203.13366](https://arxiv.org/abs/2203.13366)  
**发表**: RecSys 2022

**核心思想**:
- 将推荐任务统一为自然语言处理任务
- 所有数据（用户-item交互、用户描述、item元数据）转换为自然语言序列
- 探索了多种item索引方式：
  - **顺序索引（Sequential Indexing）**: 按顺序分配数字ID
  - **协同索引（Collaborative Indexing）**: 基于协同过滤信号
  - **语义索引（Semantic Indexing）**: 基于item内容特征
  - **混合索引（Hybrid Indexing）**: 结合多种索引方式
- 支持零样本和少样本推荐

### 2.4 M6-Rec: 阿里巴巴统一推荐基础模型

**论文**: M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems  
**链接**: [https://arxiv.org/abs/2205.08084](https://arxiv.org/abs/2205.08084)  
**发表**: arXiv 2022

**核心思想**:
- 基于阿里巴巴M6大模型构建统一推荐系统
- 将用户行为数据表示为纯文本
- 将推荐任务转换为语言理解/生成问题
- 提出改进的Prompt Tuning方法，仅使用1%的任务特定参数
- 支持检索、排序、零样本推荐、解释生成、对话推荐等多种任务

### 2.5 UniSRec: 通用序列推荐

**论文**: Towards Universal Sequence Representation Learning for Recommender Systems  
**链接**: [https://arxiv.org/abs/2206.05941](https://arxiv.org/abs/2206.05941)  
**发表**: KDD 2022

**核心思想**:
- 使用item描述文本学习可迁移的表示
- 采用参数化白化和混合专家增强的适配器
- 两个对比预训练任务学习通用序列表示
- 支持跨域、跨平台的高效迁移

---

## 3. Semantic ID的兴起（2023年）

### 3.1 TIGER: Transformer Index for Generative Recommenders

**论文**: Recommender Systems with Generative Retrieval  
**链接**: [https://arxiv.org/abs/2305.05065](https://arxiv.org/abs/2305.05065)  
**发表**: NeurIPS 2023

**核心思想**:
TIGER是生成式推荐系统的里程碑式工作，首次系统性地提出了Semantic ID的概念。

- **Semantic ID生成**: 使用RQ-VAE（Residual Quantized Variational Autoencoder）将item的dense embedding量化为离散token序列
- **层次化表示**: 通过多级量化，生成语义有意义的token元组 $(c_0, c_1, ..., c_{m-1})$
- **自回归生成**: 使用Transformer seq2seq模型逐步预测item的Semantic ID

$$
\text{RQ-VAE: } \mathbf{e} \rightarrow (c_0, c_1, ..., c_{m-1})
$$

其中每一级的code通过最小化与codebook的距离得到：

$$
c_k = \arg\min_j \| r_k - \mathbf{e}_j^{(k)} \|^2
$$

**优势**:
- 相似item共享相同的前缀token
- 更好的冷启动泛化能力
- 端到端的检索-排序统一

### 3.2 VQ-Rec: 向量量化的可迁移表示

**论文**: Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders  
**链接**: [https://arxiv.org/abs/2210.12316](https://arxiv.org/abs/2210.12316)  
**发表**: WWW 2023

**核心思想**:
- 使用向量量化学习item的离散表示
- 设计可跨任务迁移的item编码
- 为后续的向量量化推荐研究奠定了基础

### 3.3 SEATER: 语义树结构标识符

**论文**: Generative Retrieval with Semantic Tree-Structured Item Identifiers via Contrastive Learning  
**链接**: [https://paperswithcode.com/paper/generative-retrieval-with-semantic-tree](https://paperswithcode.com/paper/generative-retrieval-with-semantic-tree)  
**发表**: 2023

**核心思想**:
- 使用平衡k叉树结构的item标识符
- 每层token分配独立的语义空间
- 通过两个对比学习任务学习标识符语义：
  - **InfoNCE损失**: 基于层次位置对齐token embedding
  - **Triplet损失**: 对相似标识符进行排序
- 在多个数据集上优于SOTA模型

### 3.4 GPTRec: SVD Tokenization

**论文**: Generative Sequential Recommendation with GPTRec  
**链接**: [https://arxiv.org/abs/2306.11114](https://arxiv.org/abs/2306.11114)  
**发表**: Gen-IR@SIGIR 2023

**核心思想**:
- 基于GPT-2架构的序列推荐模型
- **SVD Tokenization**: 使用用户-item交互矩阵的SVD分解来量化item embedding，将item ID拆分为sub-id token
- embedding表大小减少40%，同时保持与SASRec相当的推荐质量
- 提出Next-K推荐策略，逐个生成推荐，考虑已推荐item的影响

### 3.5 RecFormer: 文本即一切

**论文**: Text Is All You Need: Learning Language Representations for Sequential Recommendation  
**链接**: [https://arxiv.org/abs/2305.13731](https://arxiv.org/abs/2305.13731)  
**发表**: KDD 2023

**核心思想**:
- 使用自然语言表示作为序列推荐的主要机制
- 摆脱传统的ID-based或embedding-based方法
- 通过预训练语言表示进行推荐任务的微调
- 实现跨域的有效迁移

### 3.6 TALLRec: 指令微调框架

**论文**: TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation  
**链接**: [https://arxiv.org/abs/2305.00447](https://arxiv.org/abs/2305.00447)  
**发表**: RecSys 2023

**核心思想**:
- 两阶段微调：Alpaca tuning + Rec-tuning
- 使用LoRA进行高效参数微调
- 单卡RTX 3090即可训练LLaMA-7B
- 仅需16个训练样本即可达到较好效果
- AUC从50.85提升至67.24

### 3.7 TASTE: 文本匹配序列推荐

**论文**: Text Matching Improves Sequential Recommendation by Reducing Popularity Biases  
**链接**: [https://arxiv.org/abs/2308.14029](https://arxiv.org/abs/2308.14029)  
**发表**: CIKM 2023 (Oral)

**核心思想**:
- 使用文本匹配替代传统item ID embedding
- 将item和用户-item交互文本化
- 解决ID-based系统的流行度偏差问题
- 利用预训练语言模型对长尾item进行建模
- 引入注意力稀疏方法处理长序列

### 3.8 BIGRec: 双步接地范式

**论文**: A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems  
**链接**: [https://arxiv.org/abs/2308.08434](https://arxiv.org/abs/2308.08434)  
**发表**: ACM TORS 2023

**核心思想**:
- 两步接地框架：
  1. 微调LLM生成表示item的有意义token
  2. 将生成的token映射到实际item
- 展示了LLM在推荐中的few-shot学习能力
- 发现LLM难以学习统计信息（流行度、协同过滤）

### 3.9 DreamRec: 扩散模型推荐

**论文**: Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion  
**链接**: [https://arxiv.org/abs/2310.20453](https://arxiv.org/abs/2310.20453)  
**发表**: NeurIPS 2023

**核心思想**:
- 将序列推荐重塑为learning-to-generate问题
- 使用引导扩散模型生成"oracle item"
- 从用户历史交互序列直接生成理想item表示
- 无需负采样，通过去噪扩散过程生成

---

## 4. Token化方案的深入探索（2024年）

### 4.1 LETTER: 可学习的Token化器

**论文**: LETTER: Learnable Item Tokenization for Generative Recommendation  
**链接**: [https://arxiv.org/abs/2405.07314](https://arxiv.org/abs/2405.07314)  
**发表**: CIKM 2024  
**代码**: [https://github.com/HonghuiBao2000/LETTER](https://github.com/HonghuiBao2000/LETTER)

**核心思想**:
LETTER针对现有Semantic ID方案的局限性，提出了一个综合性的解决方案：

1. **语义正则化**: 使用RQ-VAE保持层次化语义结构
2. **协同正则化**: 通过对比学习损失，引入协同过滤信号

$$
\mathcal{L}_{CL} = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\sum_{k} \exp(\text{sim}(z_i, z_k) / \tau)}
$$

3. **多样性损失**: 缓解code分配偏差，避免token塌缩
4. **排序引导生成损失**: 增强排序能力

### 4.2 TokenRec: 掩码向量量化Tokenizer

**论文**: TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation  
**链接**: [https://arxiv.org/abs/2406.10450](https://arxiv.org/abs/2406.10450)  
**发表**: 2024

**核心思想**:
- 提出**Masked Vector-Quantized (MQ) Tokenizer**
- 对掩码的用户/item表示进行量化，生成离散token
- 设计高效的生成式检索范式，减少推理时间
- 有效捕捉协同过滤知识，同时保持对新item的泛化能力

### 4.3 IDGenRec: 文本ID学习

**论文**: Towards LLM-RecSys Alignment with Textual ID Learning  
**链接**: [https://arxiv.org/abs/2403.19021](https://arxiv.org/abs/2403.19021)  
**发表**: 2024  
**代码**: [https://github.com/agiresearch/IDGenRec](https://github.com/agiresearch/IDGenRec)

**核心思想**:
- 解决LLM与推荐系统的对齐问题
- 学习将item ID转换为有意义的文本表示（Textual ID）
- 使用人类语言token作为item标识符
- 在19个数据集预训练后，在6个未见数据集上展示zero-shot能力

### 4.4 LC-Rec: 协同语义集成

**论文**: Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation  
**链接**: [https://arxiv.org/abs/2311.09049](https://arxiv.org/abs/2311.09049)  
**发表**: ICDE 2024

**核心思想**:
- 设计learning-based向量量化方法进行item索引
- 通过对齐调优任务增强协同语义集成
- 弥合LLM的语言语义与推荐系统item识别之间的差距

### 4.5 LAMIA: 多Embedding表示

**论文**: LAMIA: Learning to Learn Item Tokenization for Generative Recommendation  
**链接**: [https://arxiv.org/abs/2409.07276](https://arxiv.org/abs/2409.07276)  
**发表**: 2024

**核心思想**:
- 批判了RQ-VAE依赖单一embedding空间的局限性
- 提出学习"Item Palette"——多个独立且语义平行的embedding
- 每个embedding捕捉item的不同方面（如类别、风格、价格等）
- 通过文本重建任务进行领域特定微调

### 4.6 CLLM4Rec: 协同大语言模型

**论文**: Collaborative Large Language Model for Recommender Systems  
**链接**: [https://arxiv.org/abs/2311.01343](https://arxiv.org/abs/2311.01343)  
**发表**: WWW 2024

**核心思想**:
- 扩展LLM词表，添加特殊user/item ID token（soft token）
- **Soft+Hard Prompting策略**: 混合软token（user/item）和硬token（词表）
- 使用互正则化策略从噪声内容中提取推荐信息
- item预测头使用多项式似然，避免自回归解码的低效

### 4.7 E4SRec: 优雅高效的LLM序列推荐

**论文**: E4SRec: An Elegant Effective Efficient Extensible Solution of Large Language Models for Sequential Recommendation  
**链接**: [https://arxiv.org/abs/2312.02443](https://arxiv.org/abs/2312.02443)  
**发表**: WWW 2024

**核心思想**:
- 将LLM与ID-based推荐系统无缝集成
- 以item ID序列为输入，保证输出在候选列表内
- 单次前向传播生成整个排序列表
- 仅训练少量可插拔参数，冻结LLM

### 4.8 LLaRA: 大语言推荐助手

**论文**: LLaRA: Large Language-Recommendation Assistant  
**链接**: [https://arxiv.org/abs/2312.02445](https://arxiv.org/abs/2312.02445)  
**发表**: SIGIR 2024

**核心思想**:
- **混合Item表示**: 结合ID-based embedding和文本特征
- **适配器模块**: 桥接ID embedding和LLM输入空间
- **课程学习**: 从纯文本提示逐步过渡到混合提示
- 将用户行为序列视为新模态

### 4.9 CoLLM: 协同Embedding集成

**论文**: CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation  
**链接**: [https://arxiv.org/abs/2310.19488](https://arxiv.org/abs/2310.19488)  
**发表**: 2024

**核心思想**:
- 解决LLMRec忽视协同信息的问题
- 通过外部传统推荐模型捕获协同数据
- 将协同信息映射为"collaborative embeddings"
- 不修改LLM本身，保持灵活性

### 4.10 RA-Rec: ID表示对齐框架

**论文**: RA-Rec: An Efficient ID Representation Alignment Framework for LLM-based Recommendation  
**链接**: [https://arxiv.org/abs/2402.04527](https://arxiv.org/abs/2402.04527)  
**发表**: 2024

**核心思想**:
- 将预训练ID embedding作为soft prompt
- 创新的对齐模块
- 兼容多种ID-based方法和LLM架构
- HitRate@100提升最高3.0%，训练数据减少10倍

### 4.11 RDRec: 理由蒸馏

**论文**: RDRec: Rationale Distillation for LLM-based Recommendation  
**链接**: [https://arxiv.org/abs/2405.10587](https://arxiv.org/abs/2405.10587)  
**发表**: ACL 2024 (Short Paper)

**核心思想**:
- 使用Chain-of-Thought提示提取用户-item交互背后的理由
- 从评论中学习用户偏好和item属性的理由
- 创建更聚焦的用户和item画像
- 在Top-N和序列推荐上达到SOTA

### 4.12 PeaPOD: 个性化提示蒸馏

**论文**: PeaPOD: Personalized Prompt Distillation for Generative Recommendation  
**链接**: [https://arxiv.org/abs/2407.05033](https://arxiv.org/abs/2407.05033)  
**发表**: 2024

**核心思想**:
- 扩展提示蒸馏方法，引入个性化
- 使用注意力机制融合相似用户的embedding
- 创建个性化soft prompt
- 支持序列推荐、Top-N推荐和解释生成

### 4.13 HSTU: Meta的万亿参数推荐系统

**论文**: Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations  
**链接**: [https://arxiv.org/abs/2402.17152](https://arxiv.org/abs/2402.17152)  
**发表**: ICML 2024

**核心思想**:
HSTU（Hierarchical Sequential Transduction Units）是Meta提出的工业级推荐架构：

- **Pointwise Aggregated Attention**: 使用pointwise归一化替代softmax，适应流式数据的非稳态词表
- **高效Attention内核**: 将attention计算转换为grouped GEMMs
- **Stochastic Length (SL)**: 算法增加用户历史序列的稀疏性，降低计算成本
- 部署规模达1.5万亿参数
- 在线A/B测试提升12.4%
- 8192长度序列比FlashAttention2快5.3x-15.2x

### 4.14 Lite-LLM4Rec: 轻量级LLM推荐

**论文**: Rethinking Large Language Model Architectures for Sequential Recommendations  
**链接**: [https://arxiv.org/abs/2402.09543](https://arxiv.org/abs/2402.09543)  
**发表**: 2024

**核心思想**:
- 避免beam search解码，使用直接item投影头
- 采用层次化LLM结构
- 性能提升46.8%
- 效率提升97.28%

### 4.15 FORGE: 阿里巴巴Semantic ID工业实践

**论文**: FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets  
**链接**: [https://arxiv.org/abs/2509.20904](https://arxiv.org/abs/2509.20904)  
**发表**: EMNLP 2024 Industry

**核心思想**:
- 在淘宝140亿用户交互、2.5亿item上进行Semantic ID构建
- 探索SID构建的优化策略
- 提出两个新指标用于SID评估
- 线上测试交易量提升0.35%

### 4.16 GenRec: 直接生成推荐

**论文**: GenRec: Large Language Model for Generative Recommendation  
**链接**: [https://arxiv.org/abs/2307.00457](https://arxiv.org/abs/2307.00457)  
**发表**: ECIR 2024

**核心思想**:
- 使用原始文本数据（item名称/标题）作为ID
- 设计专门prompt增强LLM对推荐任务的理解
- 使用LoRA在LLaMA上微调
- 直接生成目标item推荐

### 4.17 LLMRec: 图增强LLM推荐

**论文**: LLMRec: Large Language Models with Graph Augmentation for Recommendation  
**链接**: [https://arxiv.org/abs/2311.00423](https://arxiv.org/abs/2311.00423)  
**发表**: WSDM 2024 (Oral)

**核心思想**:
- 将图增强与大语言模型集成
- 利用图结构信息增强推荐

### 4.18 HiD-VAE: 层次化解耦ID

**论文**: HiD-VAE: Hierarchically Disentangled Item IDs for Generative Recommendation  
**链接**: [https://arxiv.org/abs/2508.04618](https://arxiv.org/abs/2508.04618)  
**发表**: 2024

**核心思想**:
- 解决语义扁平化和ID冲突问题
- 引入层次化监督量化，对齐离散code与多级item标签
- 使用唯一性损失惩罚表示纠缠
- 提供可解释的语义路径

### 4.19 ETEGRec: 端到端生成推荐

**论文**: End-to-End Learnable Item Tokenization for Generative Recommendation  
**链接**: [https://arxiv.org/abs/2409.05546](https://arxiv.org/abs/2409.05546)  
**发表**: 2024

**核心思想**:
- 统一item tokenization和生成推荐训练
- 视为耦合过程而非分离
- 推荐导向的对齐策略：序列-item对齐、偏好-语义对齐

---

## 5. 最新进展（2025年）

### 5.1 DECOR: 分解上下文Token表示

**论文**: Learning Decomposed Contextual Token Representations from Pretrained and Collaborative Signals for Generative Recommendation  
**链接**: [https://arxiv.org/abs/2509.10468](https://arxiv.org/abs/2509.10468)  
**发表**: 2025

**核心思想**:
- 解决tokenizer预训练与推荐模型训练之间的目标不对齐问题
- 学习分解的上下文token表示
- 保留预训练语义的同时增强token的适应性
- 通过上下文化组合和embedding融合实现

### 5.2 CoFiRec: 从粗到细的Token化

**论文**: CoFiRec: Coarse-to-Fine Tokenization for Generative Recommendation  
**链接**: [https://arxiv.org/abs/2511.22707](https://arxiv.org/abs/2511.22707)  
**发表**: 2025

**核心思想**:
- **多语义层级分解**: 将item信息分解为多个语义层级
  - 高层：类别信息
  - 中层：详细描述
  - 低层：协同过滤信号
- **独立Token化**: 每个语义层级独立进行token化
- **结构化顺序**: 保持从粗到细的结构顺序
- 语言模型可以渐进式地从通用兴趣到特定item生成token

### 5.3 UTGRec: 通用Item Tokenizer

**论文**: Universal Item Tokenization for Transferable Generative Recommendation  
**链接**: [https://arxiv.org/abs/2504.04405](https://arxiv.org/abs/2504.04405)  
**发表**: 2025

**核心思想**:
- 通过适配多模态LLM（MLLM）进行通用item tokenization
- 使用树结构codebook将内容表示离散化为code
- **内容重建**: 双轻量级解码器重建item文本和图像
- **协同知识集成**: 通过共现对齐和重建集成协同信号
- 统一预训练和跨域适配框架

### 5.4 SpecGR: 归纳式生成推荐

**论文**: Inductive Generative Recommendation via Retrieval-based Speculation  
**链接**: [https://arxiv.org/abs/2410.02939](https://arxiv.org/abs/2410.02939)  
**发表**: AAAI 2026 (accepted)

**核心思想**:
- 解决生成式推荐模型如何推荐未见过的item（Unseen Items）
- 采用drafter-verifier框架
- 实现归纳式推荐能力

### 5.5 Purely Semantic Indexing

**论文**: Purely Semantic Indexing for LLM-based Generative Recommendation and Retrieval  
**链接**: [https://arxiv.org/abs/2509.16446](https://arxiv.org/abs/2509.16446)  
**发表**: 2025

**核心思想**:
- 生成唯一且语义保留的ID，无需附加非语义token
- 解决语义ID冲突问题（相似item获得相同ID）
- 两种算法：穷举候选匹配（ECM）和递归残差搜索（RRS）
- 提升整体和冷启动性能

### 5.6 PCTX: 个性化上下文感知语义ID

**论文**: Personalized Context-Aware Semantic ID Tokenization  
**链接**: [https://openreview.net/forum?id=xxx](https://openreview.net/forum?id=xxx)  
**发表**: 2025

**核心思想**:
- 解决静态语义ID tokenization的局限性
- 引入用户历史交互生成个性化语义ID
- 同一item根据用户上下文不同可以被不同地tokenize
- NDCG@10提升最高8.9%

### 5.7 HSNN: 层次化结构神经网络

**论文**: Hierarchical Structured Neural Network: Efficient Retrieval Scaling for Large Scale Recommendation  
**链接**: [https://arxiv.org/abs/2408.06653](https://arxiv.org/abs/2408.06653)  
**发表**: 2025

**核心思想**:
- 使用模块化神经网络（MoNN）学习复杂用户-item交互
- 在层次化item索引上操作，实现计算共享
- 联合学习MoNN和层次索引
- 实现亚线性计算成本

---

## 6. 综述与展望

### 6.1 重要综述论文

#### 向量量化推荐综述

**论文**: Vector Quantization for Recommender Systems: A Review and Outlook  
**链接**: [https://arxiv.org/abs/2405.03110](https://arxiv.org/abs/2405.03110)  
**发表**: 2024

该综述系统性地回顾了向量量化在推荐系统中的应用，包括：
- 三类向量量化技术
- 效率导向与质量导向的应用
- 与LLM和多模态推荐的结合

#### 生成式搜索与推荐综述

**论文**: A Survey of Generative Search and Recommendation in the Era of Large Language Models  
**链接**: [https://arxiv.org/abs/2404.16924](https://arxiv.org/abs/2404.16924)  
**发表**: 2024

- 统一框架审视生成式搜索和推荐
- 分析LLM如何以生成方式解决用户-item匹配问题

#### LLM生成式推荐综述

**论文**: Large Language Models for Generative Recommendation: A Survey and Visionary Discussions  
**链接**: [https://aclanthology.org/2024.lrec-main.886/](https://aclanthology.org/2024.lrec-main.886/)  
**发表**: LREC-COLING 2024

- 回答三个核心问题：什么是生成式推荐、为什么采用、如何实现
- 将推荐简化为直接输出推荐的单一生成阶段

### 6.2 Token化方案演进总结

| 时期 | 代表方法 | 核心特点 |
|------|----------|----------|
| 2022 | DSI, P5, M6-Rec, UniSRec | 开创性探索，文本化表示 |
| 2023 | TIGER, VQ-Rec, SEATER, TALLRec, DreamRec | Semantic ID，RQ-VAE量化，指令微调 |
| 2024 | LETTER, TokenRec, CLLM4Rec, E4SRec, LLaRA, HSTU | 多维度优化，工业级应用，协同信号集成 |
| 2025 | DECOR, CoFiRec, UTGRec, PCTX | 精细化分解，多层级表示，个性化tokenization |

### 6.3 技术路线分类

1. **基于文本的方法**: P5, RecFormer, TASTE, GenRec
   - 直接使用item文本信息
   - 利用预训练语言模型

2. **基于向量量化的方法**: TIGER, VQ-Rec, LETTER, TokenRec
   - 使用RQ-VAE等量化技术
   - 生成层次化Semantic ID

3. **混合表示方法**: LLaRA, CoLLM, CLLM4Rec, RA-Rec
   - 结合ID embedding和文本特征
   - 融合协同信号和语义信息

4. **树结构方法**: SEATER, HSNN
   - 使用层次化树结构组织item
   - 实现高效检索

5. **扩散模型方法**: DreamRec
   - 将推荐重塑为生成问题
   - 无需负采样

### 6.4 未来研究方向

1. **记忆与泛化的平衡**: 如何在区分特定item和捕捉语义相似性之间取得平衡
2. **多模态Token化**: 结合文本、图像、视频等多模态信息
3. **动态Token化**: 适应item特征的实时变化
4. **高效推理**: 降低生成式推荐的计算开销
5. **冷启动优化**: 进一步提升对新item的表示能力
6. **个性化Token化**: 根据用户上下文生成个性化的item表示
7. **工业级部署**: 解决大规模部署中的效率和稳定性问题

---

## 参考文献

1. Tay, Y., et al. "Transformer Memory as a Differentiable Search Index." NeurIPS 2022.
2. Wang, Y., et al. "A Neural Corpus Indexer for Document Retrieval." NeurIPS 2022.
3. Geng, S., et al. "Recommendation as Language Processing (RLP): P5." RecSys 2022.
4. Cui, Z., et al. "M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems." arXiv 2022.
5. Hou, Y., et al. "Towards Universal Sequence Representation Learning for Recommender Systems." KDD 2022.
6. Rajput, S., et al. "Recommender Systems with Generative Retrieval." NeurIPS 2023.
7. Hou, Y., et al. "Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders." WWW 2023.
8. Si, Z., et al. "Generative Retrieval with Semantic Tree-Structured Item Identifiers via Contrastive Learning." 2023.
9. Petrov, A., et al. "Generative Sequential Recommendation with GPTRec." Gen-IR@SIGIR 2023.
10. Li, J., et al. "Text Is All You Need: Learning Language Representations for Sequential Recommendation." KDD 2023.
11. Bao, K., et al. "TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation." RecSys 2023.
12. Lin, X., et al. "Text Matching Improves Sequential Recommendation by Reducing Popularity Biases." CIKM 2023.
13. Bao, K., et al. "A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems." ACM TORS 2023.
14. Yang, Z., et al. "Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion." NeurIPS 2023.
15. Bao, H., et al. "LETTER: Learnable Item Tokenization for Generative Recommendation." CIKM 2024.
16. Qu, W., et al. "TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation." 2024.
17. Tan, H., et al. "Towards LLM-RecSys Alignment with Textual ID Learning." 2024.
18. Ren, X., et al. "Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation." ICDE 2024.
19. Li, X., et al. "LAMIA: Learning to Learn Item Tokenization for Generative Recommendation." 2024.
20. Zhu, Y., et al. "Collaborative Large Language Model for Recommender Systems." WWW 2024.
21. Li, X., et al. "E4SRec: An Elegant Effective Efficient Extensible Solution of LLMs for Sequential Recommendation." WWW 2024.
22. Liao, J., et al. "LLaRA: Large Language-Recommendation Assistant." SIGIR 2024.
23. Zhang, Y., et al. "CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation." 2024.
24. Yang, L., et al. "RA-Rec: An Efficient ID Representation Alignment Framework for LLM-based Recommendation." 2024.
25. Wang, C., et al. "RDRec: Rationale Distillation for LLM-based Recommendation." ACL 2024.
26. Li, W., et al. "PeaPOD: Personalized Prompt Distillation for Generative Recommendation." 2024.
27. Zhai, J., et al. "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations." ICML 2024.
28. Ji, H., et al. "GenRec: Large Language Model for Generative Recommendation." ECIR 2024.
29. Wei, W., et al. "LLMRec: Large Language Models with Graph Augmentation for Recommendation." WSDM 2024.
30. Zheng, B., et al. "Universal Item Tokenization for Transferable Generative Recommendation." 2025.
31. Li, J., et al. "CoFiRec: Coarse-to-Fine Tokenization for Generative Recommendation." 2025.
