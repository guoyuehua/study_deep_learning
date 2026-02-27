import torch
import torch.nn as nn

torch.manual_seed(42)

batch_size, seq_len, feature_dim = 2, 3, 4
num_heads, head_dim = 2, 2

x = torch.randn(batch_size, seq_len, feature_dim)
print(f"输入形状: {x.shape}")
print(f"输入数据:\n{x[0]}\n")

# 方式1：一个大线性层，然后分割
print("=" * 60)
print("方式1：一个大线性层，然后分割")
print("=" * 60)

linear_q_big = nn.Linear(feature_dim, feature_dim, bias=False)
Q_big = linear_q_big(x)  # [2, 3, 4]
print(f"大线性层输出形状: {Q_big.shape}")

Q_method1 = Q_big.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
print(f"分割后形状: {Q_method1.shape}")
print(f"第1个头的Q:\n{Q_method1[0, 0]}")
print(f"第2个头的Q:\n{Q_method1[0, 1]}\n")

# 方式2：每个头独立的线性层
print("=" * 60)
print("方式2：每个头独立的线性层")
print("=" * 60)

linear_q_heads = nn.ModuleList([
    nn.Linear(feature_dim, head_dim, bias=False) for _ in range(num_heads)
])

Q_heads = []
for i in range(num_heads):
    Q_i = linear_q_heads[i](x)
    Q_heads.append(Q_i.unsqueeze(1))

Q_method2 = torch.cat(Q_heads, dim=1)
print(f"拼接后形状: {Q_method2.shape}")
print(f"第1个头的Q:\n{Q_method2[0, 0]}")
print(f"第2个头的Q:\n{Q_method2[0, 1]}\n")

# 方式3：证明等价性 - 手动设置权重
print("=" * 60)
print("方式3：证明两种方式等价（手动设置权重）")
print("=" * 60)

linear_q_big = nn.Linear(feature_dim, feature_dim, bias=False)
linear_q_head0 = nn.Linear(feature_dim, head_dim, bias=False)
linear_q_head1 = nn.Linear(feature_dim, head_dim, bias=False)

linear_q_head0.weight.data = linear_q_big.weight.data[:head_dim, :]
linear_q_head1.weight.data = linear_q_big.weight.data[head_dim:, :]

Q_big_out = linear_q_big(x)
Q_method3_from_big = Q_big_out.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

Q_head0_out = linear_q_head0(x).unsqueeze(1)
Q_head1_out = linear_q_head1(x).unsqueeze(1)
Q_method3_from_heads = torch.cat([Q_head0_out, Q_head1_out], dim=1)

print(f"方式1（大线性层分割）:\n{Q_method3_from_big[0, 0]}")
print(f"方式2（独立线性层拼接）:\n{Q_method3_from_heads[0, 0]}")
print(f"是否相等: {torch.allclose(Q_method3_from_big, Q_method3_from_heads)}\n")

# 方式4：更高效的单线性层实现
print("=" * 60)
print("方式4：更高效的单线性层实现（推荐）")
print("=" * 60)

linear_q_efficient = nn.Linear(feature_dim, num_heads * head_dim, bias=False)
Q_efficient = linear_q_efficient(x)
Q_method4 = Q_efficient.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

print(f"输出形状: {Q_method4.shape}")
print(f"参数量: {linear_q_efficient.weight.shape}\n")

# 总结
print("=" * 60)
print("总结：为什么一个大线性层有意义？")
print("=" * 60)
print("""
1. 数学等价性：
   - 一个大Linear(4, 4)等价于两个小Linear(4, 2)
   - 大线性层的权重可以看作是多个小线性层权重的拼接

2. 实现效率：
   - 一个大线性层：1次矩阵乘法
   - 多个小线性层：num_heads次矩阵乘法
   - 大线性层更高效！

3. 参数量相同：
   - 大线性层：[4, 4] = 16个参数
   - 小线性层：2 * [4, 2] = 16个参数

4. 标准实现：
   - Transformer论文使用一个大线性层
   - 然后分割成多个头
   - 这就是"多头注意力"的标准实现

5. 每个头确实有不同的变换：
   - 虽然用一个大线性层
   - 但每个头对应大线性层的不同部分
   - 相当于每个头有独立的权重子矩阵
""")
