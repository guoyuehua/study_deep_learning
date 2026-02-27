# 定义Attention类
import torch.nn as nn # 导入torch.nn库
import torch # 导入torch库


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, decoder_context, encoder_context):
        # 计算decoder_context和encoder_context的点积，得到注意力分数
        scores = torch.matmul(decoder_context, encoder_context.transpose(-2, -1))
        # 归一化分数
        attn_weights = nn.functional.softmax(scores, dim=-1)
        # 将注意力权重乘以encoder_context，得到加权的上下文向量
        context = torch.matmul(attn_weights, encoder_context)
        return context, attn_weights