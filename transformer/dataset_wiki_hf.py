from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
import torch

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for item in data_iter:
        yield tokenizer(item)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def get_data_iter(split):
    return (item['text'] for item in dataset[split])

vocab = build_vocab_from_iterator(yield_tokens(get_data_iter('train')), 
                                  specials=["<pad>", "<sos>", "<eos>"])
vocab.set_default_index(vocab["<pad>"])

print("词汇表大小：", len(vocab))
print("词汇示例(word to index): ",  {word: vocab[word] for word in ["<pad>", "<sos>", "<eos>", "the", "apple"]})

max_seq_len = 256
batch_size = 16

class WikiDataset(Dataset):
    def __init__(self, data_iter, vocab, max_len=max_seq_len):
        self.data = []        
        for sentence in data_iter:
            tokens = tokenizer(sentence)[:max_len - 2]
            tokens = [vocab["<sos>"]] + vocab(tokens) + [vocab["<eos>"]]
            self.data.append(tokens)
    
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, idx):
        source = self.data[idx][:-1]
        target = self.data[idx][1:]
        return torch.tensor(source), torch.tensor(target)

def pad_sequence(sequences, padding_value=0, length=None):
    max_length = max(len(seq) for seq in sequences) if length is None else length
    result = torch.full((len(sequences), max_length), padding_value, dtype=torch.long)    
    for i, seq in enumerate(sequences):
        end = len(seq)
        result[i, :end] = seq[:end]
    return result

def collate_fn(batch):
    sources, targets = zip(*batch)    
    max_length = max(max(len(s) for s in sources), max(len(t) for t in targets))
    sources = pad_sequence(sources, padding_value=vocab["<pad>"], length=max_length)
    targets = pad_sequence(targets, padding_value=vocab["<pad>"], length=max_length)    
    return sources, targets

train_dataset = WikiDataset(get_data_iter('train'), vocab)
print(f"Dataset数据条目数: {len(train_dataset)}")
sample_source, sample_target = train_dataset[100]
print(f"输入序列张量样例: {sample_source}")
print(f"目标序列张量样例: {sample_target}")
decoded_source = ' '.join(vocab.lookup_tokens(sample_source.tolist()))
decoded_target = ' '.join(vocab.lookup_tokens(sample_target.tolist()))
print(f"输入序列样例文本: {decoded_source}")
print(f"目标序列样例文本: {decoded_target}")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, collate_fn=collate_fn)

valid_dataset = WikiDataset(get_data_iter('validation'), vocab)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
