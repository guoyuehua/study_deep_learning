import torch
import torch.nn.functional as F
import numpy as np

print("=" * 80)
print("为什么注意力机制中对最后一维应用Softmax？")
print("=" * 80)

# 模拟注意力分数
batch_size, n_heads, len_q, len_k = 2, 2, 3, 4
scores = torch.randn(batch_size, n_heads, len_q, len_k)

print(f"\n注意力分数形状: {scores.shape}")
print(f"  - batch_size: {batch_size}")
print(f"  - n_heads: {n_heads}")
print(f"  - len_q (Query序列长度): {len_q}")
print(f"  - len_k (Key序列长度): {len_k}")

print(f"\n示例分数（batch=0, head=0）:\n{scores[0, 0]}")

# 对最后一维应用softmax
weights = F.softmax(scores, dim=-1)

print(f"\n对最后一维(dim=-1)应用Softmax后的权重:\n{weights[0, 0]}")

# 验证每行和为1
print(f"\n验证每行和为1:")
row_sums = weights[0, 0].sum(dim=-1)
print(f"每行和: {row_sums}")

print("\n" + "=" * 80)
print("详细解释：为什么是最后一维？")
print("=" * 80)

print("""
注意力分数矩阵的含义：
┌─────────────────────────────────────┐
│         Key位置 (len_k)              │
│     K1    K2    K3    K4            │
├─────────────────────────────────────┤
│ Q1 [0.2, 0.3, 0.4, 0.1]  ← Query1对各Key的注意力 │
│ Q2 [0.1, 0.5, 0.2, 0.2]  ← Query2对各Key的注意力 │
│ Q3 [0.4, 0.2, 0.3, 0.1]  ← Query3对各Key的注意力 │
│    Query位置 (len_q)                  │
└─────────────────────────────────────┘

每一行代表：
- 一个Query位置对所有Key位置的注意力分布
- 这些权重和为1（归一化）
- 表示这个Query"关注"各个Key的程度
""")

print("\n" + "=" * 80)
print("逐步演示：不同维度的Softmax效果")
print("=" * 80)

# 创建一个简单的示例
scores_simple = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],  # Query1
    [2.0, 3.0, 4.0, 5.0],  # Query2
    [3.0, 4.0, 5.0, 6.0]   # Query3
])

print(f"\n原始分数:\n{scores_simple}")

# 对最后一维应用softmax（正确）
weights_last_dim = F.softmax(scores_simple, dim=-1)
print(f"\n对最后一维(dim=-1)应用Softmax:\n{weights_last_dim}")
print(f"每行和: {weights_last_dim.sum(dim=-1)}")
print("解释：每个Query独立计算对所有Key的注意力分布")

# 对倒数第二维应用softmax（错误）
weights_second_last = F.softmax(scores_simple, dim=-2)
print(f"\n对倒数第二维(dim=-2)应用Softmax:\n{weights_second_last}")
print(f"每列和: {weights_second_last.sum(dim=-2)}")
print("解释：每个Key独立计算对所有Query的权重（不符合注意力机制）")

print("\n" + "=" * 80)
print("实际应用示例：句子翻译")
print("=" * 80)

print("""
源句子（Key）: "我 爱 学习" (3个token)
目标句子（Query）: "I love learning" (3个token)

注意力分数矩阵 [len_q=3, len_k=3]:
                我    爱   学习
    I        [0.5, 0.3, 0.2]  ← "I"主要关注"我"
    love     [0.2, 0.6, 0.2]  ← "love"主要关注"爱"
    learning [0.1, 0.2, 0.7]  ← "learning"主要关注"学习"

对最后一维Softmax：
- 每行和为1
- 每个目标词独立决定关注哪些源词
- 符合注意力机制的设计
""")

print("\n" + "=" * 80)
print("为什么不是其他维度？")
print("=" * 80)

# 创建示例
scores_example = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

print(f"示例分数 [len_q=2, len_k=3]:\n{scores_example}")

# dim=0 (第一维)
print(f"\n对dim=0应用Softmax:\n{F.softmax(scores_example, dim=0)}")
print("含义：每个Key位置对所有Query的权重（不符合注意力机制）")

# dim=1 (最后一维)
print(f"\n对dim=1(最后一维)应用Softmax:\n{F.softmax(scores_example, dim=1)}")
print("含义：每个Query位置对所有Key的权重（正确的注意力机制）")

print("\n" + "=" * 80)
print("数学原理")
print("=" * 80)

print("""
注意力机制的核心公式：
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

步骤分解：
1. Q @ K^T: 计算Query和Key的相似度
   结果形状: [batch, heads, len_q, len_k]
   
2. softmax(dim=-1): 对每个Query，归一化对所有Key的分数
   目的: 使每个Query的注意力权重和为1
   
3. @ V: 用注意力权重加权Value
   结果形状: [batch, heads, len_q, d_v]

关键理解：
- 每个Query位置需要独立决定关注哪些Key
- 因此需要在Key维度（最后一维）上归一化
- 这样每个Query的注意力分布是独立的
""")

print("\n" + "=" * 80)
print("可视化理解")
print("=" * 80)

# 创建一个更直观的例子
scores_viz = torch.tensor([
    [2.0, 1.0, 0.5, 0.1],  # Query1: 主要关注Key1
    [0.5, 2.0, 1.0, 0.5],  # Query2: 主要关注Key2
    [0.1, 0.5, 2.0, 1.0]   # Query3: 主要关注Key3
])

weights_viz = F.softmax(scores_viz, dim=-1)

print("注意力分数 → 注意力权重:")
print("\n分数矩阵:")
print("         Key1  Key2  Key3  Key4")
for i, row in enumerate(scores_viz):
    print(f"Query{i+1}  {row.tolist()}")

print("\n权重矩阵（对最后一维Softmax）:")
print("         Key1  Key2  Key3  Key4   | 和")
for i, row in enumerate(weights_viz):
    print(f"Query{i+1}  {[f'{x:.3f}' for x in row.tolist()]}  | {row.sum():.3f}")

print("\n解读：")
print("- Query1主要关注Key1 (权重最大)")
print("- Query2主要关注Key2 (权重最大)")
print("- Query3主要关注Key3 (权重最大)")
print("- 每行的权重和为1（归一化）")

print("\n" + "=" * 80)
print("总结")
print("=" * 80)

print("""
为什么对最后一维应用Softmax？

1. 维度含义：
   - 最后一维是Key的维度
   - 每个Query需要计算对所有Key的注意力分布

2. 独立性：
   - 每个Query位置独立决定关注哪些Key
   - 不同Query的注意力分布相互独立

3. 归一化：
   - 每个Query的注意力权重和为1
   - 表示注意力在不同Key上的分配比例

4. 直观理解：
   - 行：Query位置
   - 列：Key位置
   - 每行的Softmax：一个Query对所有Key的关注程度

5. 数学意义：
   - softmax(Q @ K^T)的最后一维对应Key维度
   - 对Key维度归一化，得到注意力权重
   - 用于加权求和Value

这就是为什么注意力机制中要对最后一维应用Softmax！
""")
