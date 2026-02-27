import torch
import torch.nn.functional as F
import numpy as np

print("=" * 70)
print("注意力掩码（Attention Mask）详解")
print("=" * 70)

# 示例1：简单的掩码演示
print("\n【示例1：简单的掩码演示】")
scores = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                       [2.0, 3.0, 4.0, 5.0]])
print(f"原始分数:\n{scores}")

# 创建掩码：True的位置会被填充
mask = torch.tensor([[False, False, True, True],   # 最后两个位置需要mask
                     [False, False, False, True]]) # 最后一个位置需要mask
print(f"\n掩码 (True=需要mask):\n{mask}")

# 应用掩码
scores_masked = scores.masked_fill(mask, -1e9)
print(f"\n应用掩码后:\n{scores_masked}")

# softmax
weights = F.softmax(scores_masked, dim=-1)
print(f"\nSoftmax后的权重:\n{weights}")
print("注意：被mask的位置权重接近0")

# 示例2：Transformer中的实际应用（Padding Mask）
print("\n" + "=" * 70)
print("【示例2：Padding Mask - 屏蔽填充token】")
print("=" * 70)

# 假设有一个batch的句子，长度不同
# 句子1: "我 爱 学习" (长度3)
# 句子2: "你 好" (长度2，需要padding)
seq_len = 4
batch_size = 2

# 模拟注意力分数
scores = torch.randn(batch_size, 1, seq_len, seq_len)
print(f"注意力分数形状: {scores.shape}")

# 创建padding mask
# 句子1: 前3个token有效，第4个是padding
# 句子2: 前2个token有效，后2个是padding
padding_mask = torch.tensor([
    [False, False, False, True],   # 句子1的第4个位置是padding
    [False, False, True, True]     # 句子2的第3、4个位置是padding
])
print(f"Padding mask形状: {padding_mask.shape}")

# 扩展mask维度以匹配scores
padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
print(f"扩展后的mask形状: {padding_mask.shape}")

# 应用掩码
scores_masked = scores.masked_fill(padding_mask, -1e9)
print(f"\n应用padding mask后的分数（部分）:\n{scores_masked[0, 0]}")

weights = F.softmax(scores_masked, dim=-1)
print(f"\nSoftmax后的权重（部分）:\n{weights[0, 0]}")

# 示例3：Causal Mask（因果掩码/下三角掩码）
print("\n" + "=" * 70)
print("【示例3：Causal Mask - 解码器自注意力】")
print("=" * 70)

seq_len = 5

# 创建下三角掩码（用于解码器，防止看到未来信息）
# True的位置会被mask（上三角部分）
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
print(f"Causal mask (True=需要mask):\n{causal_mask}")

# 模拟注意力分数
scores = torch.randn(1, 1, seq_len, seq_len)
print(f"\n原始分数:\n{scores[0, 0]}")

# 应用掩码
scores_masked = scores.masked_fill(causal_mask, -1e9)
print(f"\n应用causal mask后:\n{scores_masked[0, 0]}")

weights = F.softmax(scores_masked, dim=-1)
print(f"\nSoftmax后的权重:\n{weights[0, 0]}")

print("\n解释：")
print("- 第1个token只能看自己")
print("- 第2个token可以看前2个")
print("- 第3个token可以看前3个")
print("- 以此类推...")

# 示例4：为什么用 -1e9 而不是 -inf？
print("\n" + "=" * 70)
print("【示例4：为什么用 -1e9 而不是 -inf？】")
print("=" * 70)

scores1 = torch.tensor([1.0, 2.0, 3.0, -1e9])
scores2 = torch.tensor([1.0, 2.0, 3.0, float('-inf')])

weights1 = F.softmax(scores1, dim=-1)
weights2 = F.softmax(scores2, dim=-1)

print(f"使用 -1e9: {weights1}")
print(f"使用 -inf: {weights2}")
print("\n两者结果几乎相同，但 -1e9 更稳定（避免数值问题）")

# 示例5：完整的注意力计算流程
print("\n" + "=" * 70)
print("【示例5：完整的注意力计算流程】")
print("=" * 70)

batch_size, n_heads, len_q, len_k, d_k = 2, 2, 3, 4, 64

Q = torch.randn(batch_size, n_heads, len_q, d_k)
K = torch.randn(batch_size, n_heads, len_k, d_k)
V = torch.randn(batch_size, n_heads, len_k, d_k)

print(f"Q形状: {Q.shape}")
print(f"K形状: {K.shape}")
print(f"V形状: {V.shape}")

# 步骤1：计算注意力分数
scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
print(f"\n1. 注意力分数形状: {scores.shape}")

# 步骤2：创建掩码（假设最后两个位置需要mask）
attn_mask = torch.zeros(batch_size, n_heads, len_q, len_k, dtype=torch.bool)
attn_mask[:, :, :, -2:] = True  # 最后两个位置mask
print(f"2. 掩码形状: {attn_mask.shape}")

# 步骤3：应用掩码
scores_masked = scores.masked_fill(attn_mask, -1e9)
print(f"3. 应用掩码后，被mask位置的值: {scores_masked[0, 0, 0, -2:]}")

# 步骤4：softmax
weights = F.softmax(scores_masked, dim=-1)
print(f"4. Softmax后，被mask位置的权重: {weights[0, 0, 0, -2:]}")

# 步骤5：加权求和
context = torch.matmul(weights, V)
print(f"5. 上下文向量形状: {context.shape}")

print("\n" + "=" * 70)
print("总结")
print("=" * 70)
print("""
1. masked_fill_(mask, value) 的作用：
   - 将mask中True的位置填充为value
   - 下划线表示原地操作

2. 为什么填充 -1e9：
   - softmax(exp(-1e9)) ≈ 0
   - 使得被mask的位置权重接近0
   - 避免模型关注不该关注的位置

3. 常见的mask类型：
   - Padding Mask：屏蔽填充token
   - Causal Mask：屏蔽未来信息（解码器）
   - Custom Mask：自定义需要屏蔽的位置

4. 数学原理：
   softmax(x_i) = exp(x_i) / sum(exp(x_j))
   当 x_i = -1e9 时，exp(-1e9) ≈ 0
   因此该位置权重接近0，不影响其他位置
""")
