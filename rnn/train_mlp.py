import pandas_datareader as pdr
gs10 = pdr.get_data_fred('GS10')
print(gs10.head())


import matplotlib.pyplot as plt
plt.plot(gs10)
plt.show()

import torch
from torch.utils.data import DataLoader, TensorDataset

num = len(gs10)                           # 总数据量
x = torch.tensor(gs10['GS10'].to_list())  # 数据列表

seq_len = 6                               # 预测序列长度
batch_size = 4                            # 设置批大小
X_feature = torch.zeros((num - seq_len, seq_len))      # 全零初始化特征矩阵，num-seq_len行，seq_len列
for i in range(seq_len):
    X_feature[:, i] = x[i: num - seq_len + i]    # 为特征矩阵赋值
    y_label = x[seq_len:].reshape((-1, 1))       # 真实结果列表

train_loader = DataLoader(
    TensorDataset(X_feature[:num-seq_len], y_label[:num-seq_len]), 
    batch_size=batch_size, 
    shuffle=True)  # 构建数据加载器

print(train_loader.dataset[:batch_size])


from torch import nn
from tqdm import *

class Model(nn.Module):
    def __init__(self, input_size, output_size, num_hiddens):
        super().__init__()
        self.linear1 = nn.Linear(input_size, num_hiddens)
        self.linear2 = nn.Linear(num_hiddens, output_size)
    def forward(self, X):
        output = torch.relu(self.linear1(X))
        output = self.linear2(output)
        return output

# 定义超参数
input_size = seq_len
output_size = 1
num_hiddens = 10
lr = 0.01
# 构建模型
model = Model(input_size, output_size, num_hiddens)
criterion = nn.MSELoss(reduction='none')
trainer = torch.optim.Adam(model.parameters(), lr)

num_epochs = 20
loss_history = []
for epoch in tqdm(range(num_epochs)):
    # 批量训练
    for X, y in train_loader:
        trainer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.sum().backward()
        trainer.step()

     # 输出损失
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for X, y in train_loader:
            y_pred = model(X)
            loss = criterion(y_pred, y)
            total_loss += loss.sum()/loss.numel()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Validation loss = {avg_loss:.4f}')
        loss_history.append(avg_loss)
# 绘制损失和准确率的曲线图
import matplotlib.pyplot as plt
plt.plot(loss_history, label='loss')
plt.legend()
plt.show()


preds = model(X_feature)
time = torch.arange(seq_len+1, num+1, dtype= torch.float32)  # 时间轴
plt.plot(time[:num-seq_len], gs10['GS10'].to_list()[seq_len:num], label='gs10')
plt.plot(time[:num-seq_len], preds.detach().numpy(), label='preds')
plt.legend()
plt.show()