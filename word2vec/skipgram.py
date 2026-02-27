# 定义Skip-Gram类
import torch.nn as nn  # 导入neural network


class SkipGram(nn.Module):

    def __init__(self, voc_size, embedding_size):
        super(SkipGram, self).__init__()
        # 从词汇表大小到嵌入层大小（维度）的线性层（权重矩阵）
        self.input_to_hidden = nn.Linear(voc_size, embedding_size, bias=False)
        # 从嵌入层大小（维度）到词汇表大小的线性层（权重矩阵）
        self.hidden_to_output = nn.Linear(embedding_size, voc_size, bias=False)

    def forward(self, X):  # 前向传播的方式，X形状为(batch_size, voc_size)
        # 通过隐藏层，hidden形状为 (batch_size, embedding_size)
        hidden = self.input_to_hidden(X)
        # 通过输出层，output_layer形状为 (batch_size, voc_size)
        output = self.hidden_to_output(hidden)
        return output



class SkipGramV2(nn.Module):

    def __init__(self, voc_size, embedding_size):
        super(SkipGramV2, self).__init__()
        # 从词汇表大小到嵌入大小的嵌入层（权重矩阵）
        self.input_to_hidden = nn.Embedding(voc_size, embedding_size)
        # 从嵌入大小到词汇表大小的线性层（权重矩阵）
        self.hidden_to_output = nn.Linear(embedding_size, voc_size, bias=False)

    def forward(self, X):
        hidden_layer = self.input_to_hidden(X)  # 生成隐藏层：[batch_size, embedding_size]
        output_layer = self.hidden_to_output(hidden_layer)  # 生成输出层：[batch_size, voc_size]
        return output_layer
