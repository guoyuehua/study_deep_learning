import matplotlib.pyplot as plt
from utils import get_sin_enc_table
import torch
import numpy as np

pe = get_sin_enc_table(100, 128)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 完整热力图
axes[0, 0].imshow(pe.numpy(), aspect='auto', cmap='RdBu')
axes[0, 0].set_title('Full Position Encoding')
axes[0, 0].set_xlabel('Dimension')
axes[0, 0].set_ylabel('Position')

# 图2: 高频区域（前20维）
axes[0, 1].imshow(pe[:, :20].numpy(), aspect='auto', cmap='RdBu')
axes[0, 1].set_title('High Frequency (dim 0-20)')
axes[0, 1].set_xlabel('Dimension')
axes[0, 1].set_ylabel('Position')

# 图3: 低频区域（后20维）
axes[1, 0].imshow(pe[:, -20:].numpy(), aspect='auto', cmap='RdBu')
axes[1, 0].set_title('Low Frequency (dim 108-127)')
axes[1, 0].set_xlabel('Dimension')
axes[1, 0].set_ylabel('Position')

# 图4: 相似度矩阵
sim_matrix = torch.zeros(100, 100)
for i in range(100):
    for j in range(100):
        sim_matrix[i, j] = torch.cosine_similarity(
            pe[i].unsqueeze(0), pe[j].unsqueeze(0)
        ).item()
axes[1, 1].imshow(sim_matrix.numpy(), cmap='viridis')
axes[1, 1].set_title('Position Similarity Matrix')
axes[1, 1].set_xlabel('Position')
axes[1, 1].set_ylabel('Position')

plt.tight_layout()
plt.show()