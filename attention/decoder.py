import torch # 导入torch库
import torch.nn as nn # 导入torch.nn库
import torch.nn.functional as F # 导入nn.functional
import torch.optim as optim

from attention import Attention

# 定义解码器类
class DecoderWithAttention(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderWithAttention, self).__init__()
        self.hidden_size = hidden_size # 设置隐藏层大小
        self.embedding = nn.Embedding(output_size, hidden_size) # 创建词嵌入层
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True) # 创建RNN层
        self.attention = Attention()  # 创建注意力层
        self.out = nn.Linear(2 * hidden_size, output_size)  # 修改线性输出层，考虑隐藏状态和上下文向量
    def forward(self, dec_input, hidden, enc_output):
        # dec_input: [batch_size, tgt_seq_len] - 例如 [1, 4]，解码器输入序列的词索引(来自teacher forcing)
        # hidden: [num_layers, batch_size, hidden_size] - 例如 [1, 1, 128]，初始隐藏状态(来自编码器)
        # enc_output: [batch_size, src_seq_len, hidden_size] - 例如 [1, 3, 128]，编码器所有时间步的输出
        
        embedded = self.embedding(dec_input)  # [batch_size, tgt_seq_len, hidden_size] - 例如 [1, 4, 128]，将输入序列转换为嵌入向量
        
        rnn_output, hidden = self.rnn(embedded, hidden)  
        # rnn_output: [batch_size, tgt_seq_len, hidden_size] - 例如 [1, 4, 128]，每个时间步的RNN输出
        # hidden: [num_layers, batch_size, hidden_size] - 例如 [1, 1, 128]，最终的隐藏状态
        
        context, attn_weights = self.attention(rnn_output, enc_output)  
        # context: [batch_size, tgt_seq_len, hidden_size] - 例如 [1, 4, 128]，每个时间步的上下文向量
        # attn_weights: [batch_size, tgt_seq_len, src_seq_len] - 例如 [1, 4, 3]，注意力权重矩阵
        
        dec_output = torch.cat((rnn_output, context), dim=-1)  # [batch_size, tgt_seq_len, 2*hidden_size] - 例如 [1, 4, 256]，拼接RNN输出和上下文向量
        
        dec_output = self.out(dec_output)  # [batch_size, tgt_seq_len, output_size] - 例如 [1, 4, 19]，每个时间步的词汇表预测概率分布
        
        return dec_output, hidden, attn_weights
    
# 创建解码器
# decoder = DecoderWithAttention(n_hidden, voc_size_en)
# print('解码器结构：', decoder)  # 打印解码器的结构
