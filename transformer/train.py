import torch.optim as optim
import torch.nn as nn
import torch
from model import Transformer
from dataset import TranslationCorpus
from config import batch_size, epochs

sentences = [
    ['我 爱 你', 'I love you'],
    ['咖哥 喜欢 小冰', 'KaGe likes XiaoBing'],
    ['我 爱 学习 人工智能', 'I love studying AI'],
    ['深度学习 改变 世界', ' DL changed the world'],
    ['自然语言处理 很 强大', 'NLP is powerful'],
    ['神经网络 非常 复杂', 'Neural-networks are complex']
]

# 创建语料库类实例
corpus = TranslationCorpus(sentences)
model = Transformer(corpus) # 创建模型实例

criterion = nn.CrossEntropyLoss() # 损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0001) # 优化器

for epoch in range(epochs): # 训练100轮
    optimizer.zero_grad() # 梯度清零
    enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size) # 创建训练数据
    outputs, _, _, _ = model(enc_inputs, dec_inputs) # 获取模型输出 
    # 计算损失
    # outputs 原始形状: [batch_size, tgt_len, vocab_size] ，[3, 6, 20]  (假设)
    # outputs.view(-1, vocab_size): [batch_size × tgt_len, vocab_size]， [18, 20]
    # target_batch 原始形状: [batch_size, tgt_len]， [3, 6]
    # target_batch.view(-1): [batch_size × tgt_len]， [18]
    # CrossEntropyLoss 要求:
    # - 预测: [N, C] (N个样本，C个类别)
    # - 目标: [N] (N个类别索引)
    loss = criterion(outputs.view(-1, len(corpus.tgt_vocab)), target_batch.view(-1)) # 计算损失
    if (epoch + 1) % 1 == 0: # 打印损失
        print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")
    
    loss.backward()# 反向传播        
    optimizer.step()# 更新参数


def greedy_decode(model, corpus, enc_inputs, max_len):
    """贪心解码：自回归生成"""
    batch_size = enc_inputs.size(0)
    
    # 编码器前向传播（只需一次）
    enc_outputs, _ = model.encoder(enc_inputs)
    
    # 初始化解码器输入：只有 <sos>
    dec_inputs = torch.zeros(batch_size, max_len).long()
    dec_inputs[:, 0] = corpus.tgt_vocab['<sos>']
    
    # 逐位置生成
    for i in range(max_len - 1):
        # 解码器前向传播
        dec_outputs, _, _ = model.decoder(
            dec_inputs[:, :i+1], enc_inputs, enc_outputs)
        
        # 获取下一个词的预测
        proj_outputs = model.projection(dec_outputs)
        next_word = proj_outputs[:, -1, :].max(dim=-1)[1]
        
        # 将预测词加入输入
        dec_inputs[:, i+1] = next_word
        
        # 如果预测到 <eos>，提前结束
        if next_word.item() == corpus.tgt_vocab['<eos>']:
            break
    
    return dec_inputs

enc_inputs, _, _ = corpus.make_batch(batch_size=1, test_batch=True)

max_len = corpus.tgt_len
dec_outputs = greedy_decode(model, corpus, enc_inputs, max_len)

translated_sentence = []
for idx in dec_outputs[0]:
    word = corpus.tgt_idx2word[idx.item()]
    if word == '<eos>':
        break
    if word not in ['<sos>', '<pad>']:
        translated_sentence.append(word)

input_sentence = ' '.join([corpus.src_idx2word[idx.item()] for idx in enc_inputs[0]])
print(input_sentence, '->', translated_sentence)

