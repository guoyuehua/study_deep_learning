# 构建语料库，每行包含中文、英文（解码器输入）和翻译成英文后的目标输出3个句子
sentences = [
    ['我 爱 你', '<sos> I love you', 'I love you <eos>'],
    ['咖哥 喜欢 小冰', '<sos> KaGe likes XiaoBing', 'KaGe likes XiaoBing <eos>'],
    ['我 爱 学习 人工智能', '<sos> I love studying AI', 'I love studying AI <eos>'],
    ['深度学习 改变 世界', '<sos> DL changed the world', 'DL changed the world <eos>'],
    ['自然 语言 处理 很 强大', '<sos> NLP is so powerful', 'NLP is so powerful <eos>'],
    ['神经网络 非常 复杂', '<sos> Neural-Nets are complex', 'Neural-Nets are complex <eos>']
]
word_list_cn, word_list_en = [], []  # 初始化中英文词汇表
# 遍历每一个句子并将单词添加到词汇表中
for s in sentences:
    word_list_cn.extend(s[0].split())
    word_list_en.extend(s[1].split())
    word_list_en.extend(s[2].split())

# 去重，得到没有重复单词的词汇表
word_list_cn = list(set(word_list_cn))
word_list_en = list(set(word_list_en))

# 构建单词到索引的映射
word2idx_cn = {w: i for i, w in enumerate(word_list_cn)}
word2idx_en = {w: i for i, w in enumerate(word_list_en)}

# 构建索引到单词的映射
idx2word_cn = {i: w for i, w in enumerate(word_list_cn)}
idx2word_en = {i: w for i, w in enumerate(word_list_en)}

# 计算词汇表的大小
voc_size_cn = len(word_list_cn)
voc_size_en = len(word_list_en)
print("句子数量：", len(sentences)) # 打印句子数量
print("中文词汇表大小：", voc_size_cn) #打印中文词汇表大小
print("英文词汇表大小：", voc_size_en) #打印英文词汇表大小
print("中文词汇到索引的字典：", word2idx_cn) # 打印中文词汇到索引的字典
print("英文词汇到索引的字典：", word2idx_en) # 打印英文词汇到索引的字典


import numpy as np # 导入numpy
import torch # 导入torch
import random # 导入random库
# 定义一个函数，随机选择一个句子和词汇表生成输入、输出和目标数据
def make_data(sentences):
    # 随机选择一个句子进行训练
    random_sentence = random.choice(sentences)
    # 将输入句子中的单词转换为对应的索引
    encoder_input = np.array([[word2idx_cn[n] for n in random_sentence[0].split()]])
    # 将输出句子中的单词转换为对应的索引
    decoder_input = np.array([[word2idx_en[n] for n in random_sentence[1].split()]])
    # 将目标句子中的单词转换为对应的索引
    target = np.array([[word2idx_en[n] for n in random_sentence[2].split()]])

    # 将输入、输出和目标批次转换为LongTensor
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input)
    target = torch.LongTensor(target)
    return encoder_input, decoder_input, target


# 推荐的做法
def make_batch(sentences, batch_size=4):
    # 1. 随机选择batch_size个句子
    batch_sentences = random.sample(sentences, batch_size)
    
    # 2. 找到最大长度
    max_len_cn = max(len(s[0].split()) for s in batch_sentences)
    max_len_en = max(len(s[1].split()) for s in batch_sentences)
    
    # 3. Padding到相同长度
    encoder_inputs = []
    decoder_inputs = []
    targets = []
    
    for s in batch_sentences:
        cn_indices = [word2idx_cn[w] for w in s[0].split()]
        cn_indices += [0] * (max_len_cn - len(cn_indices))
        encoder_inputs.append(cn_indices)
        
        en_indices = [word2idx_en[w] for w in s[1].split()]
        en_indices += [0] * (max_len_en - len(en_indices))
        decoder_inputs.append(en_indices)
        
        target_indices = [word2idx_en[w] for w in s[2].split()]
        target_indices += [0] * (max_len_en - len(target_indices))
        targets.append(target_indices)
    
    return torch.LongTensor(encoder_inputs), torch.LongTensor(decoder_inputs), torch.LongTensor(targets)


#使用make_data函数生成输入、输出和目标张量
encoder_input, decoder_input, target = make_data(sentences)
for s in sentences: # 获取原始句子
    if all([word2idx_cn[w] in encoder_input[0] for w in s[0].split()]):
        original_sentence = s
        break

print("原始句子：", original_sentence) # 打印原始句子
print("编码器输入张量的形状：", encoder_input.shape)  # 打印输入张量形状
print("解码器输入张量的形状：", decoder_input.shape)  # 打印输出张量形状
print("目标张量的形状：", target.shape) # 打印目标张量形状
print("编码器输入张量：", encoder_input) # 打印输入张量
print("解码器输入张量：", decoder_input) # 打印输出张量
print("目标张量：", target) # 打印目标张量

import torch.nn as nn # 导入torch.nn库
# 定义编码器类，继承自nn.Module
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()       
        self.hidden_size = hidden_size # 设置隐藏层大小       
        self.embedding = nn.Embedding(input_size, hidden_size) # 创建词嵌入层       
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True) # 创建RNN层
    def forward(self, inputs, hidden): # 前向传播函数
        # inputs: [batch, seq_len] - 例如 [1, 3]，输入序列的词索引
        # hidden: [num_layers, batch, hidden_size] - 例如 [1, 1, 128]，初始隐藏状态
        embedded = self.embedding(inputs) # 将输入转换为嵌入向量
        # embedded: [batch, seq_len, hidden_size] - 例如 [1, 3, 128]，词嵌入向量
        output, hidden = self.rnn(embedded, hidden) # 将嵌入向量输入RNN层并获取输出
        # output: [batch, seq_len, hidden_size] - 例如 [1, 3, 128]，每个时间步的隐藏状态输出
        # hidden: [num_layers, batch, hidden_size] - 例如 [1, 1, 128]，最终的隐藏状态
        return output, hidden

# 定义解码器类，继承自nn.Module
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()       
        self.hidden_size = hidden_size # 设置隐藏层大小
        self.embedding = nn.Embedding(output_size, hidden_size) # 创建词嵌入层
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)  # 创建RNN层
        self.out = nn.Linear(hidden_size, output_size) # 创建线性输出层    
    def forward(self, inputs, hidden):  # 前向传播函数
        # inputs: [batch, seq_len] - 例如 [1, 1] 或 [1, 4]，输入序列的词索引
        # hidden: [num_layers, batch, hidden_size] - 例如 [1, 1, 128]，初始隐藏状态（来自编码器）
        embedded = self.embedding(inputs) # 将输入转换为嵌入向量
        # embedded: [batch, seq_len, hidden_size] - 例如 [1, 1, 128] 或 [1, 4, 128]，词嵌入向量
        output, hidden = self.rnn(embedded, hidden) # 将嵌入向量输入RNN层并获取输出
        # output: [batch, seq_len, hidden_size] - 例如 [1, 1, 128] 或 [1, 4, 128]，每个时间步的隐藏状态输出
        # hidden: [num_layers, batch, hidden_size] - 例如 [1, 1, 128]，更新后的隐藏状态
        output = self.out(output) # 使用线性层生成最终输出
        # output: [batch, seq_len, output_size] - 例如 [1, 1, 19] 或 [1, 4, 19]，词汇表预测
        return output, hidden
    

n_hidden = 128 # 设置隐藏层数量
# 创建编码器和解码器
encoder = Encoder(voc_size_cn, n_hidden)
decoder = Decoder(n_hidden, voc_size_en)
print('编码器结构：', encoder)  # 打印编码器的结构
print('解码器结构：', decoder)  # 打印解码器的结构


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        # 初始化编码器和解码器
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, enc_input, hidden, dec_input):    # 定义前向传播函数
        # 使输入序列通过编码器并获取输出和隐藏状态
        encoder_output, encoder_hidden = self.encoder(enc_input, hidden)
        # 将编码器的隐藏状态传递给解码器作为初始隐藏状态
        decoder_hidden = encoder_hidden
        # 使解码器输入（目标序列）通过解码器并获取输出
        decoder_output, _ = self.decoder(dec_input, decoder_hidden)
        return decoder_output
    
# 创建Seq2Seq架构
model = Seq2Seq(encoder, decoder)
print('S2S模型结构：', model)  # 打印模型的结构


# 定义训练函数
def train_seq2seq(model, criterion, optimizer, epochs):
    for epoch in range(epochs):
       encoder_input, decoder_input, target = make_data(sentences) # 训练数据的创建
       hidden = torch.zeros(1, encoder_input.size(0), n_hidden) # 初始化隐藏状态
       optimizer.zero_grad()# 梯度清零        
       output = model(encoder_input, hidden, decoder_input) # 获取模型输出        
       loss = criterion(output.view(-1, voc_size_en), target.view(-1)) # 计算损失
       if (epoch + 1) % 40 == 0: # 打印损失
          print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")         
       loss.backward()# 反向传播        
       optimizer.step()# 更新参数

# 训练模型
epochs = 400 # 训练轮次
criterion = nn.CrossEntropyLoss() # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 优化器
train_seq2seq(model, criterion, optimizer, epochs) # 调用函数训练模型


# 定义测试函数
def test_seq2seq(model, source_sentence):
    # 将输入的句子转换为索引
    encoder_input = np.array([[word2idx_cn[n] for n in source_sentence.split()]])
    # 构建输出的句子的索引，以'<sos>'开始，后面跟'<eos>'，长度与输入句子相同
    decoder_input = np.array([word2idx_en['<sos>']] + [word2idx_en['<eos>']]*(len(encoder_input[0])-1))
    # 转换为LongTensor类型
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input).unsqueeze(0) # 增加一维    
    hidden = torch.zeros(1, encoder_input.size(0), n_hidden) # 初始化隐藏状态    
    predict = model(encoder_input, hidden, decoder_input) # 获取模型输出    
    predict = predict.data.max(2, keepdim=True)[1] # 获取概率最大的索引
    # 打印输入的句子和预测的句子
    print(source_sentence, '->', [idx2word_en[n.item()] for n in predict.squeeze()])

# 测试模型
test_seq2seq(model, '咖哥 喜欢 小冰')  
test_seq2seq(model, '自然 语言 处理 很 强大')


# 训练时使用teacher forcing，测试时使用逐步生成
def train_seq2seq_better(model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        encoder_input, decoder_input, target = make_data(sentences)
        # encoder_input: [1, seq_len_cn] - 例如 [1, 3]，编码器输入（中文句子索引）
        # decoder_input: [1, seq_len_en] - 例如 [1, 4]，解码器输入（英文句子索引，带<sos>）
        # target: [1, seq_len_en] - 例如 [1, 4]，目标输出（英文句子索引，带<eos>）
        
        hidden = torch.zeros(1, encoder_input.size(0), n_hidden)
        # hidden: [1, 1, 128] - [num_layers, batch, hidden_size]，初始隐藏状态
        
        optimizer.zero_grad()
        
        # 逐步生成，而不是一次性
        encoder_output, encoder_hidden = model.encoder(encoder_input, hidden)
        # encoder_output: [1, seq_len_cn, 128] - [batch, seq_len, hidden_size]，编码器每个时间步的输出
        # encoder_hidden: [1, 1, 128] - [num_layers, batch, hidden_size]，编码器最终隐藏状态
        
        decoder_hidden = encoder_hidden
        # decoder_hidden: [1, 1, 128] - [num_layers, batch, hidden_size]，解码器初始隐藏状态（来自编码器）
        
        outputs = []
        # outputs: [] - 空列表，后续会存储每个时间步的输出张量
        
        decoder_input_current = decoder_input[:, :1]
        # decoder_input_current: [1, 1] - [batch, 1]，只取第一个词<sos>作为初始输入
        
        for t in range(decoder_input.size(1) - 1):
            # 循环次数: seq_len_en - 1，例如 4-1=3 次
            decoder_output, decoder_hidden = model.decoder(decoder_input_current, decoder_hidden)
            # decoder_output: [1, 1, 19] - [batch, 1, voc_size_en]，当前时间步的词汇表预测
            # decoder_hidden: [1, 1, 128] - [num_layers, batch, hidden_size]，更新后的隐藏状态
            outputs.append(decoder_output)
            # outputs: [[1, 1, 19], [1, 1, 19], ...] - 列表包含每个时间步的输出
            
            # Teacher forcing: 使用真实的目标词作为下一个输入
            decoder_input_current = decoder_input[:, t+1:t+2]
            # decoder_input_current: [1, 1] - [batch, 1]，取下一个真实目标词作为输入
            # t=0时取decoder_input[:, 1:2]（第2个词），t=1时取decoder_input[:, 2:3]（第3个词），以此类推
        
        output = torch.cat(outputs, dim=1)
        # output: [1, seq_len_en-1, 19] - [batch, seq_len-1, voc_size_en]，拼接所有时间步的输出
        # 例如: [1, 3, 19]，表示3个时间步的预测，每个时间步预测19个词汇
        
        loss = criterion(output.view(-1, voc_size_en), target[:, 1:].view(-1))
        # output.view(-1, voc_size_en): [seq_len_en-1, 19] - 例如 [3, 19]，展平的预测（CrossEntropyLoss要求2D）
        # target[:, 1:]: [1, seq_len_en-1] - 例如 [1, 3]，去掉第一个词<sos>的目标
        # target[:, 1:].view(-1): [seq_len_en-1] - 例如 [3]，展平的目标（CrossEntropyLoss要求1D）
        # loss: scalar - 标量损失值
        
        loss.backward()
        # 反向传播，计算梯度
        
        optimizer.step()
        # 更新模型参数

def test_seq2seq_better(model, source_sentence, max_length=20):
    encoder_input = np.array([[word2idx_cn[n] for n in source_sentence.split()]])
    encoder_input = torch.LongTensor(encoder_input)
    
    hidden = torch.zeros(1, encoder_input.size(0), n_hidden)
    
    # 编码
    encoder_output, encoder_hidden = model.encoder(encoder_input, hidden)
    
    # 逐步解码
    decoder_input = torch.LongTensor([[word2idx_en['<sos>']]])
    decoder_hidden = encoder_hidden
    
    result = []
    for _ in range(max_length):
        decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
        predict = decoder_output.argmax(dim=-1)
        
        predicted_word = idx2word_en[predict[0, 0].item()]
        result.append(predicted_word)
        
        if predicted_word == '<eos>':
            break
        
        decoder_input = predict
    
    print(source_sentence, '->', result)