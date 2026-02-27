from torchtext.datasets import WikiText2 # 导入WikiText2
from torchtext.data.utils import get_tokenizer # 导入Tokenizer分词工具
from torchtext.vocab import build_vocab_from_iterator # 导入Vocabulary工具
from torch.utils.data import DataLoader, Dataset # 导入Pytorch的DataLoader和Dataset

tokenizer = get_tokenizer("basic_english") # 定义数据预处理所需的Tokenizer

def yield_tokens(data_iter):
    for item in data_iter:
        yield tokenizer(item)

train_iter_for_vocab = WikiText2(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter_for_vocab), 
                                  specials=["<pad>", "<sos>", "<eos>"])
train_iter = WikiText2(split='train')
vocab.set_default_index(vocab["<pad>"])
# 打印词汇表信息
print("词汇表大小：", len(vocab))
print("词汇示例(word to index): ",  {word: vocab[word] for word in ["<pad>", "<sos>", "<eos>", "the", "apple"]})

import torch
from torch.utils.data import Dataset # 导入Dataset类
max_seq_len = 256 # 设置序列的最大长度
batch_size = 16

# 定义一个处理WikiText2数据集的自定义数据集类
class WikiDataset(Dataset):
    def __init__(self, data_iter, vocab, max_len=max_seq_len):
        self.data = []        
        for sentence in data_iter: # 遍历数据集，将文本转换为tokens
            # 对每个句子进行Tokenization，截取长度为max_len-2，为<sos>和<eos>留出空间
            tokens = tokenizer(sentence)[:max_len - 2]
            tokens = [vocab["<sos>"]] + vocab(tokens) + [vocab["<eos>"]] # 添加<sos>和<eos>
            self.data.append(tokens) # 将处理好的tokens添加到数据集中
    
    def __len__(self): # 定义数据集的长度
        return len(self.data)    
    
    def __getitem__(self, idx): # 定义数据集的索引方法 （即抽取数据条目）        
        source = self.data[idx][:-1] # 获取当前数据，并将<eos>移除，作为源(source)数据
        target = self.data[idx][1:] # 获取当前数据，并将<sos>移除，作为目标(target)数据（右移1位）       
        return torch.tensor(source), torch.tensor(target) # 转换为tensor并返回


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


train_dataset = WikiDataset(train_iter, vocab) # 创建训练数据集
print(f"Dataset数据条目数: {len(train_dataset)}")
sample_source, sample_target = train_dataset[100]
print(f"输入序列张量样例: {sample_source}")
print(f"目标序列张量样例: {sample_target}")
decoded_source = ' '.join(vocab.lookup_tokens(sample_source.tolist()))
decoded_target = ' '.join(vocab.lookup_tokens(sample_target.tolist()))
print(f"输入序列样例文本: {decoded_source}")
print(f"目标序列样例文本: {decoded_target}")

# 创建一个训练数据加载器，使用自定义的collate_fn函数
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, collate_fn=collate_fn)


# …… 之前的代码（加载WikiText2数据集）
valid_iter = WikiText2(split='valid')  # 加载WikiText2数据集的验证部分
# ……之前的代码（创建数据集）
valid_dataset = WikiDataset(valid_iter, vocab)  # 创建验证数据集
# ……之前的代码（创建数据加载器）
# 创建一个验证数据加载器，使用自定义的collate_fn函数
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn) 