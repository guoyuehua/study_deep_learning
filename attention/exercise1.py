import torch
import torch.nn.functional as F

# 一个形状为 (batch_size, seq_len, feature_dim) 的张量 x
x = torch.randn(2, 3, 4)  # 形状 (batch_size, seq_len, feature_dim) 
# 定义头数和每个头的维度
num_heads = 2
head_dim = 2
# feature_dim 必须是 num_heads * head_dim 的整数倍
assert x.size(-1) == num_heads * head_dim

# 定义线性层用于将 x 转换为 Q, K, V 向量
linear_q = torch.nn.Linear(4, 4)
linear_k = torch.nn.Linear(4, 4)
linear_v = torch.nn.Linear(4, 4)
# 通过线性层计算 Q, K, V
Q = linear_q(x)  # 形状 (batch_size, seq_len, feature_dim) 
K = linear_k(x)  # 形状 (batch_size, seq_len, feature_dim) 
V = linear_v(x)  # 形状 (batch_size, seq_len, feature_dim) 

# 将 Q, K, V 分割成 num_heads 个头
def split_heads(tensor, num_heads):
    batch_size, seq_len, feature_dim = tensor.size()
    head_dim = feature_dim // num_heads
    output = tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    return  output # 形状 (batch_size, num_heads, seq_len, header_dim)

Q = split_heads(Q, num_heads)  # 形状 (batch_size, num_heads, seq_len, head_dim)
K = split_heads(K, num_heads)  # 形状 (batch_size, num_heads, seq_len, head_dim)
V = split_heads(V, num_heads)  # 形状 (batch_size, num_heads, seq_len, head_dim)

# 计算 Q 和 K 的点积，作为相似度分数，也就是自注意力原始权重
raw_weights = torch.matmul(Q, K.transpose(-2, -1))  # 形状 (batch_size, num_heads, seq_len, seq_len)
# 对自注意力原始权重进行缩放
scale_factor = K.size(-1) ** 0.5
scaled_weights = raw_weights / scale_factor  # 形状 (batch_size, num_heads, seq_len, seq_len)
# 用softmax函数对缩放后的权重进行归一化，得到注意力权重
attn_weights = F.softmax(scaled_weights, dim=-1)  # 形状 (batch_size, num_heads, seq_len, seq_len)
# 将注意力权重应用于 V 向量，计算加权和，得到加权信息
attn_outputs = torch.matmul(attn_weights, V)  # 形状 (batch_size, num_heads, seq_len, head_dim)
# 将所有头的结果拼接起来
def combine_heads(tensor, num_heads):
    batch_size, num_heads, seq_len, head_dim = tensor.size()
    feature_dim = num_heads * head_dim
    output = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, feature_dim)
    return output # 形状: (batch_size, seq_len, feature_dim)

attn_outputs = combine_heads(attn_outputs, num_heads)  # 形状 (batch_size, seq_len, feature_dim) 
# 对拼接后的结果进行线性变换
linear_out = torch.nn.Linear(4, 4)
attn_outputs = linear_out(attn_outputs)  # 形状 (batch_size, seq_len, feature_dim) 
print("加权信息：", attn_outputs)
