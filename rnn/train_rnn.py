import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas_datareader as pdr

gs10 = pdr.get_data_fred('GS10')
print(gs10.head(20))

num = len(gs10)  # 总数据量，59
x = torch.tensor(gs10['GS10'].to_list())  # 数据列表
seq_len = 6  # 预测序列长度
batch_size = 4  # 设置批大小

X_feature = torch.zeros((num - seq_len, seq_len))  # 构建特征矩阵，num-seq_len行，seq_len列，初始值均为0
Y_label = torch.zeros((num - seq_len, seq_len))  # 构建标签矩阵，形状同特征矩阵

for i in range(seq_len):
    X_feature[:, i] = x[i: num - seq_len + i]  # 为特征矩阵赋值
    Y_label[:, i] = x[i + 1: num - seq_len + i + 1]  # 为标签矩阵赋值

train_loader = DataLoader(
    TensorDataset(
        X_feature[:num - seq_len].unsqueeze(2),
        Y_label[:num - seq_len]
    ),
    batch_size=batch_size, shuffle=True)  # 构建数据加载器

from torch import nn
from tqdm import *


class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, num_hiddens, n_layers):
        super(RNNModel, self).__init__()
        self.num_hiddens = num_hiddens
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, num_hiddens, n_layers, batch_first=True)
        self.linear = nn.Linear(num_hiddens, output_size)

    def forward(self, X):
        batch_size = X.size(0)
        state = self.begin_state(batch_size)
        output, state = self.rnn(X, state)
        output = self.linear(torch.relu(output))
        return output, state

    def begin_state(self, batch_size=1):
        return torch.zeros(self.n_layers, batch_size, self.num_hiddens)


# 定义超参数
input_size = 1
output_size = 1
num_hiddens = 10
n_layers = 1
lr = 0.01
# 构建模型
model = RNNModel(input_size, output_size, num_hiddens, n_layers)
criterion = nn.MSELoss(reduction='none')
trainer = torch.optim.Adam(model.parameters(), lr)


num_epochs = 20
rnn_loss_history = []
for epoch in tqdm(range(num_epochs)):
    # 批量训练
    for X, Y in train_loader:
        trainer.zero_grad()
        y_pred, state = model(X)
        loss = criterion(y_pred.squeeze(), Y.squeeze())
        loss.sum().backward()
        trainer.step()
     # 输出损失
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for X, Y in train_loader:
            y_pred, state = model(X)
            loss = criterion(y_pred.squeeze(), Y.squeeze())
            total_loss += loss.sum()/loss.numel()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Validation loss = {avg_loss:.4f}')
        rnn_loss_history.append(avg_loss)

# 绘制损失曲线图
import matplotlib.pyplot as plt
from train_mlp import loss_history

plt.plot(loss_history, label='loss')
plt.plot(rnn_loss_history, label='RNN_loss')
plt.legend()
plt.show()
