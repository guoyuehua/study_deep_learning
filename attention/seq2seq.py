from torch import nn

# 定义Seq2Seq类
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        # 初始化编码器和解码器
        self.encoder = encoder
        self.decoder = decoder    
    def forward(self, encoder_input, hidden, decoder_input):
        # encoder_input: [batch_size, src_seq_len] - 例如 [1, 3]，源语言输入序列的词索引
        # hidden: [num_layers, batch_size, hidden_size] - 例如 [1, 1, 128]，编码器初始隐藏状态(通常为零)
        # decoder_input: [batch_size, tgt_seq_len] - 例如 [1, 4]，目标语言输入序列的词索引(用于teacher forcing)
        
        # 将输入序列通过编码器并获取输出和隐藏状态
        encoder_output, encoder_hidden = self.encoder(encoder_input, hidden)
        # encoder_output: [batch_size, src_seq_len, hidden_size] - 例如 [1, 3, 128]，编码器所有时间步的输出(用于注意力机制)
        # encoder_hidden: [num_layers, batch_size, hidden_size] - 例如 [1, 1, 128]，编码器最终的隐藏状态
        
        # 将编码器的隐藏状态传递给解码器作为初始隐藏状态
        decoder_hidden = encoder_hidden
        # decoder_hidden: [num_layers, batch_size, hidden_size] - 例如 [1, 1, 128]，解码器的初始隐藏状态
        
        # 将目标序列通过解码器并获取输出，此处更新解码器调用
        decoder_output, _, attn_weights = self.decoder(decoder_input, decoder_hidden, encoder_output)
        # decoder_output: [batch_size, tgt_seq_len, output_size] - 例如 [1, 4, 19]，每个时间步的词汇表预测概率
        # attn_weights: [batch_size, tgt_seq_len, src_seq_len] - 例如 [1, 4, 3]，注意力权重矩阵(可视化对齐关系)
        return decoder_output, attn_weights

# # 创建Seq2Seq模型
# model = Seq2Seq(encoder, decoder)
# print（'S2S模型结构：', model）  # 打印模型的结构