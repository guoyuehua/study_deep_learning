import torch #导入torch
from torch.utils.data import Dataset #导入Dataset
from torchtext.datasets import WikiText2 # 导入WikiText2
from torchtext.data.utils import get_tokenizer # 导入Tokenizer分词工具
from torchtext.vocab import build_vocab_from_iterator # 导入Vocabulary工具
tokenizer = get_tokenizer("basic_english") # 定义数据预处理所需的tokenizer
train_iter = WikiText2(split='train') # 加载WikiText2数据集的训练部分
# 定义一个生成器函数，用于将数据集中的文本转换为token
def yield_tokens(data_iter):
    for item in data_iter:
        yield tokenizer(item)
# 创建词汇表，包括特殊token: "<pad>", "<sos>", "<eos>"
vocab = build_vocab_from_iterator(yield_tokens(train_iter), 
                                  specials=["<pad>", "<sos>", "<eos>"])
vocab.set_default_index(vocab["<pad>"])


class ChatDataset(Dataset):
    def __init__(self, file_path, tokenizer, vocab):
        self.tokenizer = tokenizer #分词器
        self.vocab = vocab #词汇表
        self.input_data, self.target_data = self.load_and_process_data(file_path)
    def load_and_process_data(self, file_path):        
        with open(file_path, "r") as f:
            lines = f.readlines() # 打开文件，读取每一行数据

        input_data, target_data = [], []
        for i, line in enumerate(lines):
            if line.startswith("User:"): # 移除 "User: " 前缀，构建输入序列
                tokens = self.tokenizer(line.strip()[6:])  
                tokens = ["<sos>"] + tokens + ["<eos>"]
                indices = [self.vocab[token] for token in tokens]
                input_data.append(torch.tensor(indices, dtype=torch.long))
            elif line.startswith("AI:"): # 移除 "AI: " 前缀，构建目标序列
                tokens = self.tokenizer(line.strip()[4:])  
                tokens = ["<sos>"] + tokens + ["<eos>"]
                indices = [self.vocab[token] for token in tokens]
                target_data.append(torch.tensor(indices, dtype=torch.long))
        return input_data, target_data
    def __len__(self): #数据集的长度
        return len(self.input_data) 
    def __getitem__(self, idx): #根据索引获取数据样本
        return self.input_data[idx], self.target_data[idx] 

file_path = "chat.txt" # 加载chat.txt语料库
chat_dataset = ChatDataset(file_path, tokenizer, vocab)

for i in range(3): # 打印几个样本数据
    input_sample, target_sample = chat_dataset[i]
    print(f"Sample {i + 1}:")
    print("Input Data: ", input_sample)
    print("Target Data: ", target_sample)
    print("-" * 50)


from torch.utils.data import DataLoader # 导入DataLoader
# 定义pad_sequence函数，用于将一批序列补齐到相同长度
def pad_sequence(sequences, padding_value=0, length=None):
    # 计算最大序列长度，如果length参数未提供，则使用输入序列中的最大长度
    max_length = max(len(seq) for seq in sequences) if length is None else length
    # 创建一个具有适当形状的全零张量，用于存储补齐后的序列
    result = torch.full((len(sequences), max_length), padding_value, dtype=torch.long)    
    # 遍历序列，将每个序列的内容复制到张量result中
    for i, seq in enumerate(sequences):
        end = len(seq)
        result[i, :end] = seq[:end]
    return result

# 定义collate_fn函数，用于将一个批次的数据整理成适当的形状
def collate_fn(batch):
    # 从批次中分离源序列和目标序列
    sources, targets = zip(*batch)    
    # 计算批次中的最大序列长度
    max_length = max(max(len(s) for s in sources), max(len(t) for t in targets))
    # 使用pad_sequence函数补齐源序列和目标序列
    sources = pad_sequence(sources, padding_value=vocab["<pad>"], length=max_length)
    targets = pad_sequence(targets, padding_value=vocab["<pad>"], length=max_length)    
    # 返回补齐后的源序列和目标序列
    return sources, targets

# 创建DataLoader
batch_size = 2
chat_dataloader = DataLoader(chat_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
