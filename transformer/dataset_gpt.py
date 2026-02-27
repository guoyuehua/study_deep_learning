import torch
# 构建语料库
from collections import Counter

class LanguageCorpus:
    def __init__(self, sentences):
          self.sentences = sentences
          # 计算语言的最大句子长度，并加2以容纳特殊符号<sos>和<eos>
          self.seq_len = max([len(sentence.split()) for sentence in sentences]) + 2
          self.vocab = self.create_vocabulary() # 创建源语言和目标语言的词汇表
          self.idx2word = {v: k for k, v in self.vocab.items()} # 创建索引到单词的映射
    def create_vocabulary(self):
          vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
          counter = Counter()
          # 统计语料库的单词频率
          for sentence in self.sentences:
               words = sentence.split()
               counter.update(words)
          # 创建词汇表，并为每个单词分配一个唯一的索引
          for word in counter:
               if word not in vocab:
                  vocab[word] = len(vocab)
          return vocab
    def make_batch(self, batch_size, test_batch=False):
          input_batch, output_batch = [], [] # 初始化批次数据
          sentence_indices = torch.randperm(len(self.sentences))[:batch_size] # 随机选择句子索引
          for index in sentence_indices:
              sentence = self.sentences[index]
              # 将句子转换为索引序列
              seq = [self.vocab['<sos>']] + [self.vocab[word] for word in sentence.split()] + [self.vocab['<eos>']]
              seq += [self.vocab['<pad>']] * (self.seq_len - len(seq)) # 对序列进行填充
              # 将处理好的序列添加到批次中
              input_batch.append(seq[:-1])
              output_batch.append(seq[1:])
          return torch.LongTensor(input_batch), torch.LongTensor(output_batch)