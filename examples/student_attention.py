"""注意力机制的实现：使用 PyTorch 手写缩放点积注意力。"""

import torch


def my_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Q: [B,H,Tq,D]，K/V: [B,H,Tk,D]，输出: [B,H,Tq,D]。

    输入为已经生成的 Q、K、V；本例不包含线性投影、mask 或 dropout。
    """
    # 1. 交换 K 的最后两个维度：[B,H,Tk,D] → [B,H,D,Tk]。
    k_transposed = k.transpose(-2, -1)

    # 2. 每个 query 与每个 key 做点积，得到 [B,H,Tq,Tk] 的分数。
    scores = torch.matmul(q, k_transposed)

    # 3. 分数除以每个注意力头特征维度的平方根，控制分数尺度。
    head_dim = q.shape[-1]
    scores = scores / (head_dim ** 0.5)

    # 4. 沿 key 维度进行 softmax，每个 query 对应的权重之和为 1。
    weights = torch.softmax(scores, dim=-1)

    # 5. 根据权重汇总各个 key 对应的 value，输出 [B,H,Tq,D]。
    output = torch.matmul(weights, v)
    return output
