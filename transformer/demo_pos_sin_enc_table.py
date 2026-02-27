import matplotlib.pyplot as plt
import numpy as np
from utils import get_sin_enc_table
import torch

# 生成位置编码
pe = get_sin_enc_table(100, 128)

# 可视化
plt.figure(figsize=(10, 6))
plt.imshow(pe.numpy(), aspect='auto', cmap='RdBu')
plt.xlabel('Embedding Dimension')
plt.ylabel('Position')
plt.title('Sinusoidal Position Encoding')
plt.colorbar()
plt.show()

# 验证相对位置关系
pos_0 = pe[0]
pos_5 = pe[5]
pos_10 = pe[10]

# 相似度分析
print(f"pos0-pos5 相似度: {torch.cosine_similarity(pos_0.unsqueeze(0), pos_5.unsqueeze(0)).item():.4f}")
print(f"pos0-pos10 相似度: {torch.cosine_similarity(pos_0.unsqueeze(0), pos_10.unsqueeze(0)).item():.4f}")




def visualize_rotation_planes():
    d = 8  # 4个平面
    positions = [0, 1, 5, 10, 20]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for plane_idx, ax in enumerate(axes.flat):
        omega = 1 / np.power(10000, 2 * plane_idx / d)
        
        circle = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(circle), np.sin(circle), 'gray', alpha=0.3)
        
        for i, pos in enumerate(positions):
            theta = pos * omega
            x = np.cos(theta)
            y = np.sin(theta)
            ax.scatter([x], [y], c=colors[i], s=100, label=f'pos={pos}')
            ax.arrow(0, 0, x*0.9, y*0.9, head_width=0.1, 
                    head_length=0.05, fc=colors[i], ec=colors[i], alpha=0.5)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(f'Plane {plane_idx} (dim {2*plane_idx}, {2*plane_idx+1})\nω={omega:.4f}')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.show()

visualize_rotation_planes()