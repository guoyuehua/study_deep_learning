# study_deep_learning

深度学习学习与实践项目，包含RNN、Transformer、GPT等主流模型的实现与训练。

## 项目结构

```
study_deep_learning/
├── attention/         # 注意力机制相关实现
├── rnn/               # RNN模型实现
├── seq2seq/           # 序列到序列模型
├── transformer/       # Transformer和GPT模型
│   ├── dataset_wiki_hf.py  # WikiText2数据集处理（使用Hugging Face）
│   ├── model_gpt.py        # GPT模型实现
│   └── train_wiki.py       # 训练脚本
├── word2vec/          # Word2Vec模型
├── environment.yml    # Conda环境配置
├── requirements.txt   # 依赖包列表
└── README.md          # 项目说明文档
```

## 主要功能

- **RNN模型**：基础循环神经网络实现
- **Transformer**：完整的Transformer架构实现
- **GPT模型**：基于Transformer的生成式预训练模型
- **注意力机制**：各种注意力机制的实现和演示
- **Word2Vec**：词向量模型（CBOW和Skip-Gram）
- **序列到序列**：机器翻译等任务的实现

## 环境要求

- Python 3.10
- PyTorch 2.1.1+ 
- torchtext 0.16.1+
- 其他依赖见 `requirements.txt`

## 安装部署

### 方法1：使用Conda环境（推荐）

1. **创建并激活环境**
   ```bash
   conda env create -f environment.yml
   conda activate py3_10
   ```

2. **安装额外依赖**
   ```bash
   pip install datasets
   ```

### 方法2：使用pip安装

1. **创建虚拟环境**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # venv\Scripts\activate  # Windows
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   pip install datasets
   ```

## 快速开始

### 1. 训练GPT模型（WikiText2数据集）

```bash
# 使用Hugging Face datasets加载数据（推荐）
python transformer/train_wiki.py

# 或者使用原始torchtext加载数据
# 修改train_wiki.py中的导入语句为：from dataset_wiki import ...
```

### 2. 模型生成文本

训练完成后，脚本会自动加载最佳模型并生成文本：

```python
input_str = "my name"
generated_text = generate_text_beam_search(model, input_str)
print(f"生成的文本：{generated_text}")
```

## 模型架构

### GPT模型

- **编码器**：多层Transformer解码器
- **注意力机制**：多头自注意力
- **位置编码**：正弦位置编码
- **词嵌入**：可训练的词向量
- **输出层**：线性层映射到词汇表

### 训练参数

- **批量大小**：64
- **最大序列长度**：256
- **学习率**：0.0001
- **优化器**：Adam
- **损失函数**：CrossEntropyLoss（忽略padding）
- **梯度裁剪**：max_norm=1.0

## 数据集

- **WikiText2**：英文维基百科文本，用于语言模型训练
- **自定义对话数据**：支持聊天机器人训练（见 `dataset_chat.py`）

## 监控与评估

训练过程中会输出详细的监控信息：

- **批次级监控**：损失、梯度范数、耗时
- **Epoch级统计**：平均/最小/最大损失、训练/验证耗时
- **模型保存**：保存验证损失最低的模型

## 常见问题

### 1. 哈希校验错误

如果遇到 `RuntimeError: The computed hash ... does not match the expected hash` 错误：
- 原因：torchtext/torchdata版本兼容性问题
- 解决方案：使用 `dataset_wiki_hf.py`（基于Hugging Face datasets）

### 2. 内存不足

- 减少批量大小（`batch_size`）
- 减小最大序列长度（`max_seq_len`）
- 使用梯度累积

### 3. 生成文本质量差

- 增加训练轮次（`epochs`）
- 调整学习率
- 使用更大的数据集

## 目录说明

| 目录 | 主要功能 | 文件说明 |
|------|----------|----------|
| attention/ | 注意力机制 | 包含各种注意力实现和演示 |
| rnn/ | 循环神经网络 | 基础RNN模型实现 |
| seq2seq/ | 序列到序列 | 机器翻译等任务 |
| transformer/ | Transformer模型 | 包含完整的Transformer和GPT实现 |
| word2vec/ | 词向量模型 | CBOW和Skip-Gram实现 |

## 扩展与定制

1. **添加新数据集**：
   - 在 `transformer/` 目录创建新的数据集处理文件
   - 实现类似 `dataset_wiki_hf.py` 的接口

2. **修改模型架构**：
   - 编辑 `model_gpt.py` 调整模型参数
   - 修改 `config.py` 中的配置

3. **添加新功能**：
   - 实现不同的注意力机制
   - 添加新的生成策略（如Top-k采样）

## 依赖版本

| 包 | 版本 | 用途 |
|-----|------|------|
| torch | 2.1.1 | 深度学习框架 |
| torchtext | 0.16.1 | 文本处理 |
| datasets | 4.5.0 | 数据集加载 |
| pandas | 2.3.3 | 数据处理 |
| numpy | 1.26.4 | 数值计算 |
| matplotlib | 3.10.3 | 可视化 |

## 许可证

本项目仅供学习和研究使用。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [PyTorch官方文档](https://pytorch.org/docs/stable/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
