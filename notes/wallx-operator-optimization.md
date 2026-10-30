# Wall-X 算子优化：融合、访存与数据布局

Wall-X 的算子优化主要落在四件事上：合并相邻计算、减少显存往返、利用固定的 token 布局，以及让矩阵形状更适合 GPU 执行。判断一项优化时，需要同时看数学是否保持一致、少了哪些读写或启动，以及它在哪些输入上生效。

本文整理源码审阅中确认的机制，覆盖 Qwen2.5 与 Qwen3.5 的部分模型路径。Wall-X 负责模型调用与算子接入，许多底层 CUDA C++ 内核由 xcompute 提供；没有对应实现时保留 PyTorch fallback。不同模型路径和安装版本支持的算子不同，不能把下表理解为所有模型默认同时启用了全部优化。

文中的伪代码用于解释计算关系，不是内部内核源码，也不是可直接替换模型的实现。本文不附未经统一基准验证的加速倍数。

## 1. 优化点总览

| 优化点 | 原来的计算方式 | 优化机制 | 主要收益来源 |
| --- | --- | --- | --- |
| gate/up 投影合并 | 两次独立矩阵乘法 | 一次 GEMM 生成两个投影 | 减少 GEMM 启动，改善矩阵执行形状 |
| SiLU × Mul 融合 | 激活、乘法分别执行 | 一个 kernel 直接生成乘积 | 少写读一次激活中间结果 |
| RMSNorm 系列 | 加法、归约、缩放、门控分散执行 | 向量化访存、寄存器复用、相邻操作融合 | 减少数据遍历与调度，适用反向路径复用统计量 |
| RoPE 融合 | Q/K 分开处理，partial rotary 再拼接 | 合并旋转、复用位置表、融合尾部复制 | 减少重复读取与中间缓冲 |
| QKV 后处理融合 | 拆分、归一化、动作行替换分别执行 | 合并后处理并输出目标布局 | 减少独立 kernel 和临时张量 |
| 固定双专家重排 | 通用排序后 gather/scatter | 用连续动作区间与 offsets 直接 pack/unpack | 省去通用排序与排序索引元数据 |
| Attention 原生 GQA | 显式复制 K/V 到 Q 的头数 | kernel 直接接受不同的 Q/KV 头数 | 避免扩展 K/V 的物化 |
| GEMM 维度对齐 | 中间维度不利于硬件分块 | 补零到更合适的维度 | 改善 tile 利用率，以额外算量换执行效率 |

这些优化主要改变执行方式。乘加总量基本不变的融合也可能更快；为了对齐而略增计算量，也可能缩短实际耗时。

## 2. SwiGLU：投影合并与激活融合是两层优化

设输入为 `[M, d]`，其中 `M = batch × sequence`，FFN 中间维度为 `F`。SwiGLU 的核心关系为：

```text
gate = Linear_gate(x)                   # [M, F]
up   = Linear_up(x)                     # [M, F]
h    = SiLU(gate) × up                  # [M, F]
y    = Linear_down(h)                   # [M, d]
```

### 2.1 两次投影合成一次 GEMM

将 gate 与 up 的权重沿输出维拼接，一次矩阵乘法得到 `[M, 2F]`，再按约定访问两半。

```python
# 原来的计算关系
gate = linear(x, W_gate)
up = linear(x, W_up)
h = silu(gate) * up

# 合并后的计算关系
gate_up = linear(x, concat_output_dim(W_gate, W_up))
h = fused_silu_mul(gate_up)
```

PyTorch Linear 的权重布局为 `[out_features, in_features]`，因此权重沿 `dim=0` 拼接；输出张量则沿最后一维拆分。这两个维度不能混淆，参见 [PyTorch Linear 文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Linear.html)。

投影合并没有消除原本的乘加运算，但减少了一次 GEMM 调度，并可能获得更合适的执行形状。旧 checkpoint 可以在加载时合并权重，运行时不必每次重新拼接。

### 2.2 SiLU 与乘法放进同一个 kernel

SiLU 为 `g × sigmoid(g)`。融合内核读取 gate 和 up，直接计算 `gate × sigmoid(gate) × up`，不必先把完整的 SiLU 输出写回显存，再由乘法内核读出。

下面是仅考虑这两个逐元素操作、忽略缓存命中及反向保存的简化流量分析：

| 执行方式 | 读取 | 写入 |
| --- | --- | --- |
| 分开执行 | gate、SiLU 中间结果、up | SiLU 中间结果、最终结果 |
| 融合执行 | gate、up | 最终结果 |

若每个张量含 `M × F` 个元素，理想化的流量从 `5MF` 降到 `3MF` 个元素。这个比例描述局部读写量，不是实测加速比，也不包含前后的 GEMM。

所审阅的内核还会根据地址、stride 和维度对齐条件选择不同宽度的向量访存，最高使用 16 字节访问。一次搬运多个相邻元素，有助于减少访存指令开销。

训练时必须保留反向需要的输入。即便接口传入 `inplace=True`，有梯度的 split 路径仍使用反向安全的输出分配；不能把推理中的原地写回泛化为训练中完全零分配。

## 3. RMSNorm：归约、寄存器复用与相邻融合

RMSNorm 沿隐藏维计算：

```text
rstd = 1 / sqrt(mean(x²) + eps)
y = x × rstd × weight
```

这里没有 LayerNorm 的减均值步骤。RMSNorm 的算法背景可参考 [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)。本文讨论的是既定 RMSNorm 数学下的执行优化，不是任意替换模型中的归一化类型。

### 3.1 单算子内部怎样优化

在满足条件的宽访存路径中，内核将输入读入寄存器，用 FP32 累积平方和，通过 warp 内 shuffle 与 block 级归约得到每行统计量，然后复用寄存器中的输入完成归一化和缩放。

这条路径减少了输入的重复全局读取，也把平方、归约、倒平方根、缩放组织在同一内核中。向量宽度和寄存器占用需要按隐藏维度与数据类型选择；输入不满足对齐条件时使用通用路径。

普通 RMSNorm 的训练实现会保存前向的每行 `rstd`，反向直接复用，避免重新计算同一份平方和。保存少量统计量，可以换掉反向的一次归约。

### 3.2 AddRMSNorm：融合残差加法

```text
s = residual + x
y = RMSNorm(s)
返回 (s, y)
```

加法与归一化合并后，可以减少两个独立算子之间的启动和数据往返。残差和 `s` 仍要输出，因为后面的残差连接还需要它；不能把收益描述为完全删除这个结果。

Qwen3.5 的相关路径还要保留 zero-centered 权重语义，即有效缩放为 `1 + weight`，而不是直接使用原始 weight。

### 3.3 GatedRMSNorm：融合门控

```text
y = RMSNorm(x) × SiLU(gate)
```

将归一化、权重缩放与门控放在一起，减少中间结果的完整写回和重读。

反向统计量复用有具体条件：所审阅版本中，AddRMSNorm 的该快路径要求残差和的梯度存在，且隐藏维度不超过 4096；GatedRMSNorm 的专用统计量复用路径针对隐藏维度 128。其他情况仍有通用实现，不能声称所有 norm 的所有反向形状都采用同一条快路径。

## 4. RoPE：旋转与数据搬运一起考虑

RoPE 的基本关系可以写成：

```text
rotated = x × cos + rotate(x) × sin
```

直接展开时，旋转、乘法、加法和拼接容易形成多个小操作。可用的融合内核在一次调用中处理 Q/K，并让多个 head 复用位置表；所审阅的 CUDA 实现还利用 shared memory 和向量化访问组织这段计算。

有两个值得单独区分的接口形态：

- **packed QKV**：直接在约定的 QKV 缓冲中访问 Q/K 并旋转，避免为了这个操作额外拆出、拼回大张量。
- **partial rotary 的完整输出**：一部分 head 维度做旋转，其余维度保持原值。完整输出内核把旋转部分与不旋转尾部一起写出，省去后续 `cat` 的一次处理。

实际启用依赖算子版本、输入布局及旋转维度。half-split 与 interleaved 两种旋转约定、partial rotary 的范围，以及多模态位置轴都必须保持一致；布局相同大小并不代表语义相同。

## 5. QKV 后处理：融合拆分、归一化与动作行替换

在 Qwen3.5 的一条双专家动作路径上，上下文与动作专家分别产生投影结果。之后还需要完成：

1. 按约定拆出 Q、gate、K、V。
2. 对上下文与动作专家的 Q/K 做对应的 zero-centered RMSNorm。
3. 将动作专家结果写入每个样本的动作 token 区间。
4. 输出注意力后端需要的布局。

把这段后处理组合成一个算子，可以减少分散的 kernel launch、归一化中间结果，以及独立 clone/替换带来的搬运。

融合边界必须说清楚：这条路径前面的投影 GEMM 和投影结果拼接仍然存在，后续 transpose、RoPE 也仍然执行。因此它是 **QKV 后处理融合**，不能称为 QKV 全链路一次内核。

该路径依赖动作布局、action embeddings、CUDA 与对应的 xcompute 内核。条件不满足时使用普通处理或 PyTorch fallback；归一化本身的数学计算没有被省略。

## 6. 固定双专家：用布局信息替代通用排序

通用 MoE 需要按 expert assignment 重排 token，再送入不同专家。若只存在上下文与动作两个固定专家，并且每个样本的动作 token 是一个连续区间，就可以直接根据区间与预计算 offsets 定位。

```text
原布局：每个样本中混合上下文区与动作区
                       ↓ 按区间直接 pack
专家布局：[全部上下文 token][全部动作 token]
                       ↓ 两段连续的专家 MLP
                       ↓ unpack
恢复原 batch / sequence 布局
```

优化省去的是通用排序和排序索引元数据。token 本身仍然需要读写，pack/unpack 不是零拷贝。如果上游已使用一致的专家布局，本层还能跳过重复的 pack/unpack。

适用条件是固定两专家、动作区间连续、offsets 与原序列对应。它不能直接推广成任意 top-k 稀疏 MoE；PyTorch fallback 中的逐 batch 循环也不具有专用 CUDA 内核的效率。

## 7. Attention：减少中间矩阵与 K/V 复制

这里有两层不同的优化。

**第一层是注意力内核本身。** FlashAttention 通过分块与片上数据复用减少 HBM 读写，避免把完整 attention score / probability 矩阵存入显存。它保持标准 attention 的数学关系，dense attention 的计算量仍随序列长度呈二次增长。原理见 [FlashAttention 论文](https://arxiv.org/abs/2205.14135)。

**第二层是原生 GQA 输入。** 设 Q 有 `Hq` 个头，K/V 有 `Hkv` 个头，且 `Hkv < Hq`。如果后端支持这两种头数，可以直接消费：

```text
Q: [B, S, Hq,  Dh]
K: [B, S, Hkv, Dh]
V: [B, S, Hkv, Dh]
```

无需先把 K/V 显式复制到 `Hq` 个头。相对于复制后的 K/V 缓冲，这部分元素数保持在 `Hkv / Hq`；这个比例不是整层显存或整模型耗时的同比下降。GQA 的模型背景见 [GQA 论文](https://arxiv.org/abs/2305.13245)。

Wall-X 所审阅的一条 FlashMask 路径使用原生 GQA，并按 KV 头数处理 mask。它要求后端支持 Q/KV 不同头数；旧路径仍可能使用 K/V 重复。wrapper 中的 transpose、contiguous 和必要的 dtype 转换也仍然存在。

## 8. GEMM 对齐：多算一点，也可能更快

矩阵乘法耗时受 tile、对齐、数据类型和矩阵长宽比例影响，不能只由 FLOPs 判断。

所审阅的 Qwen2.5 MLP 把中间维度向上补到 128 的倍数，例如：

```text
原中间维度 F = 3420
对齐后维度 F' = ceil(3420 / 128) × 128 = 3456
```

加载 checkpoint 时，对 gate/up 的输出维与 down 的输入维补零；有效通道的计算关系保持不变。保存时裁去 padding，保留原始 checkpoint 的形状约定。

这里增加了约 1.05% 的中间宽度及对应计算、存储，换取可能更好的 GEMM 分块效率。能否缩短耗时必须在目标硬件和实际 shape 上测量，不能把某个源码注释里的 MFU 数字当作通用收益。

## 9. 正确性与性能怎样验证

### 9.1 正确性先覆盖前向和反向

对融合内核与独立 PyTorch reference 使用相同输入与上游梯度，比较输出、输入梯度和参数梯度。按 FP32、FP16、BF16 分别设定有依据的误差容限，避免用放宽容差掩盖布局或约定错误。

有代表性的边界包括：

- 空动作区间、不同 batch 的动作起止位置、pack/unpack 往返。
- Q/KV 头数不同、mask 与 KV 头数对应、非连续输入。
- zero-centered norm、不同 eps、极小值及较大数值。
- partial rotary 与旋转约定、未旋转尾部是否保持原值。
- 训练与推理、有无梯度、activation checkpointing 重算时是否改坏输入。
- 旧 checkpoint 加载合并、padding，以及保存后的形状兼容。

### 9.2 单算子收益需要回到整层验证

| 层级 | 建议记录的指标 | 目的 |
| --- | --- | --- |
| 单算子 | forward / backward GPU 时间、启动数量、峰值临时内存 | 确认融合真正省掉了什么 |
| 整个 block | 总耗时、布局转换与拷贝占比、重计算成本 | 排除局部加速被额外操作抵消 |
| 训练或推理链路 | step time / 请求延迟、吞吐、峰值显存 | 判断用户可感知的最终收益 |

计时前完成预热，使用 CUDA Event 等正确处理异步执行的计时方法。记录 GPU、库版本、shape、dtype、layout、是否训练及是否开启 CUDA Graph；固定这些条件后再比较结果。可分别测量 eager 与 Graph 场景，避免把 Graph 减少提交开销的效果误归给算子本身。

本文完成的是机制梳理与适用条件核对，没有给出一组统一环境下的新性能测量结果。后续加入 benchmark 时，应同时提供基线、输入条件、误差与端到端影响。

## 10. 相关阅读

- [Harrix 机器人推理优化详解](./robot-inference-optimization.md)：把单算子放回 prefill、KV cache 与动作去噪链路。
- [VGA 推理加速](./vga-inference-acceleration.md)：区分算子选择、CUDA Graph、Stream 与缓存各自优化的边界。
- [机器人动作 Tokenizer 与多模态训练详解](./action-tokenizer-and-multimodal-training.md)：理解动作表示与模型输入协议。
