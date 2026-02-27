import torch
import torch.nn as nn
from model_gpt import GPT
from dataset_gpt import LanguageCorpus
from config import batch_size

with open("lang.txt", "r") as file: # 从文件中读入语料
    sentences = [line.strip() for line in file.readlines()]

corpus = LanguageCorpus(sentences) # 创建语料库
vocab_size = len(corpus.vocab) # 词汇表大小
max_seq_len = corpus.seq_len # 最大句子长度（用于设置位置编码）
print(f"语料库词汇表大小: {vocab_size}") # 打印词汇表大小
print(f"最长句子长度: {max_seq_len}") # 打印最大序列长度


import torch.optim as optim # 导入优化器
device = "cuda" if torch.cuda.is_available() else "cpu" # 设置设备
model = GPT(vocab_size, max_seq_len).to(device) # 创建GPT模型实例
criterion = nn.CrossEntropyLoss() # 损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0001) # 优化器

epochs = 500 # 训练轮次
for epoch in range(epochs):  # 训练epochs轮
    optimizer.zero_grad() # 梯度清零
    inputs, targets = corpus.make_batch(batch_size) # 创建训练数据
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs) # 获取模型输出 
    loss = criterion(outputs.view(-1, vocab_size), targets.view(-1)) # 计算损失
    if (epoch + 1) % 100 == 0: # 打印损失
        print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")
    loss.backward() # 反向传播
    optimizer.step() # 更新参数

# 测试文本生成
def generate_text(model, input_str, max_len=50):
    model.eval()  # 将模型设置为评估（测试）模式，关闭dropout和batch normalization等训练相关的层
    # 将输入字符串中的每个token 转换为其在词汇表中的索引
    input_tokens = [corpus.vocab[token] for token in input_str]
    # 创建一个新列表，将输入的tokens复制到输出tokens中，目前只有输入的词
    output_tokens = input_tokens.copy()
    with torch.no_grad():  # 禁用梯度计算，以节省内存并加速测试过程
        for _ in range(max_len):  # 生成最多max_len个tokens
            # 将输出的token转换为 PyTorch张量，并增加一个代表批次的维度[1, len(output_tokens)]
            inputs = torch.LongTensor(output_tokens).unsqueeze(0).to(device)
            outputs = model(inputs) #输出 logits形状为[1, len(output_tokens), vocab_size]
            # 在最后一个维度上获取logits中的最大值，并返回其索引（即下一个token）
            _, next_token = torch.max(outputs[:, -1, :], dim=-1)            
            next_token = next_token.item() # 将张量转换为Python整数            
            if next_token == corpus.vocab["<eos>"]:
                break # 如果生成的token是 EOS（结束符），则停止生成过程           
            output_tokens.append(next_token) # 将生成的tokens添加到output_tokens列表
    # 将输出tokens转换回文本字符串
    output_str = " ".join([corpus.idx2word[token] for token in output_tokens])
    return output_str

input_str = ["Python"] # 输入一个词：Python
generated_text = generate_text(model, input_str) # 模型根据这个词生成后续文本
print("生成的文本：", generated_text) # 打印预测文本
