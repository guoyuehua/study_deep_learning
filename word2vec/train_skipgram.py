# 定义一个句子列表，后面会用这些句子来训练CBOW和Skip-Gram模型
sentences = ["Kage is Teacher",
             "Mazong is Boss",
             "Niuzong is Boss",
             "Xiaobing is Student",
             "Xiaoxue is Student"
             ]
# 将所有句子连接在一起，然后用空格分隔成多个单词
words = '  '.join(sentences).split()
# 构建词汇表，去除重复的词
word_list = list(set(words))
# 创建一个字典，将每个词映射到一个唯一的索引
word_to_idx = {word: idx for idx, word in enumerate(word_list)}
# 创建一个字典，将每个索引映射到对应的词
idx_to_word = {idx: word for idx, word in enumerate(word_list)}
voc_size = len(word_list)  # 计算词汇表的大小
print("词汇表：", word_list)  # 输出词汇表
print("词汇到索引的字典：", word_to_idx)  # 输出词汇到索引的字典
print("索引到词汇的字典：", idx_to_word)  # 输出索引到词汇的字典
print("词汇表大小：", voc_size)  # 输出词汇表大小


# 生成Skip-Gram训练数据
def create_skipgram_dataset(sentences, window_size=2):
    data = []  # 初始化数据
    for sentence in sentences:  # 遍历句子
        sentence = sentence.split()  # 将句子分割成单词列表
        for idx, word in enumerate(sentence):  # 遍历单词及其索引
            # 获取相邻的单词，将当前单词前后各N个单词作为相邻单词
            for neighbor in sentence[max(idx - window_size, 0): min(idx + window_size + 1, len(sentence))]:
                if neighbor != word:  # 排除当前单词本身
                    # 将相邻单词与当前单词作为一组训练数据
                    data.append((word, neighbor))
    return data


# 使用函数创建Skip-Gram训练数据
skipgram_data = create_skipgram_dataset(sentences)
# 打印未编码的Skip-Gram数据样例（前3个）
print("Skip-Gram数据样例（未编码）：", skipgram_data[:20])


# 定义One-Hot编码函数
import torch  # 导入torch库


def one_hot_encoding(word, word_to_idx):
    tensor = torch.zeros(len(word_to_idx))  # 创建一个长度与词汇表相同的全0张量
    tensor[word_to_idx[word]] = 1  # 将对应词索引位置上的值设为1
    return tensor  # 返回生成的One-Hot编码后的向量


# 展示One-Hot编码前后的数据
word_example = "Teacher"
print("One-Hot编码前的单词：", word_example)
print("One-Hot编码后的向量：", one_hot_encoding(word_example, word_to_idx))
# 展示编码后的Skip-Gram训练数据样例
print("Skip-Gram 数据样例（已编码）：",
      [(one_hot_encoding(target, word_to_idx), word_to_idx[context]) for context, target in skipgram_data[:3]])

import torch.nn as nn
from skipgram import SkipGram

embedding_size = 2  # 设定嵌入层的大小，这里选择2是为了方便展示 
skipgram_model = SkipGram(voc_size, embedding_size)  # 实例化Skip-Gram模型

print("Skip-Gram类：", skipgram_model)

# 训练Skip-Gram类
learning_rate = 0.001  # 设置学习速率
epochs = 1000  # 设置训练轮次
criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
import torch.optim as optim  # 导入随机梯度下降优化器

optimizer = optim.Adam(skipgram_model.parameters(), lr=learning_rate)
# 开始训练循环
loss_values = []  # 用于存储每轮的平均损失值
for epoch in range(epochs):
    loss_sum = 0  # 初始化损失值

    for center_word, context in skipgram_data:
        X = one_hot_encoding(center_word, word_to_idx).float().unsqueeze(0)  # 将中心词转换为 One-Hot 向量

        # 使用 SkipGramV2 时，不需要将中心词转为one-hot向量
        # X = torch.tensor([word_to_idx[target]], dtype=torch.long)  # 输入是中心词

        y_pred = skipgram_model(X)  # 计算预测值

        y_true = torch.tensor([word_to_idx[context]], dtype=torch.long)  # 将周围词转换为索引值

        loss = criterion(y_pred, y_true)  # 计算损失

        loss_sum += loss.item()  # 累积损失
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    if (epoch + 1) % 100 == 0:  # 输出每100轮的损失，并记录损失
        print(f"Epoch: {epoch + 1}, Loss: {loss_sum / len(skipgram_data)}")
        loss_values.append(loss_sum / len(skipgram_data))

# 绘制训练损失曲线
import matplotlib.pyplot as plt  # 导入matplotlib

# 绘制二维词向量图
plt.rcParams["font.family"] = ['SimHei']  # 用来设定字体样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来设定无衬线字体样式
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.plot(range(1, epochs // 100 + 1), loss_values)  # 绘图
plt.title('训练损失曲线')  # 图题
plt.xlabel('轮次')  # X轴Label
plt.ylabel('损失')  # Y轴Label
plt.show()  # 显示图

# 获取训练后的词向量
word_vectors = skipgram_model.input_to_hidden.weight.data.numpy()

# 绘制二维词向量
plt.figure(figsize=(10, 8))
for word, idx in word_to_idx.items():
    plt.scatter(word_vectors[0, idx], word_vectors[1, idx])
    plt.annotate(word, (word_vectors[0, idx], word_vectors[1, idx]))
plt.title('二维词向量可视化')
plt.grid(True)
plt.show()