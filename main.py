import random
import torch
import numpy as np
import spacy
import pandas as pd
import torch.nn as nn
import math
import copy

from zhconv import convert
from collections import Counter
from tqdm import tqdm
from sacrebleu import corpus_bleu
from torch.nn.functional import log_softmax

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

"""数据导入和预处理
    函数：
        seq_padding : 批次划分，数据填充
        cht_to_chs  : 将数据的繁体转化为简体
        tokenize_en : 英文分词器
        tokenize_cn : 中文分词器
    类：
        PrepareData : 数据预处理
        Batch       : 掩码类
"""
PAD = 0
UNK = 1
# 导入分词器
spacy_en = spacy.load('en_core_web_sm')
spacy_zh = spacy.load('zh_core_web_sm')


def seq_padding(X, padding = PAD):
    """
    按批次对数据填充、长度对齐
    """
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

# 繁体 ➡️ 简体
def cht_to_chs(sent):
    sent = convert(sent, "zh-cn")
    sent.encode("utf-8")
    return sent

# 定义分词函数
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_cn(text):
    return [tok.text for tok in spacy_zh.tokenizer(text)]

class PrepareData:
    def __init__(self, data_file, train_ratio, batch_size):
        # 读词，分词
        self.en_sents, self.cn_sents = self.load_data(data_file)
        # 构建词表
        self.en_vocab, self.en_idx2word = self.build_vocab(self.en_sents)
        self.cn_vocab, self.cn_idx2word = self.build_vocab(self.cn_sents)
        # 编码句子
        self.en_encode, self.cn_encode = self.encode_sentence(self.en_sents, self.cn_sents, self.en_vocab, self.cn_vocab, sort=False)
        # 数据分离
        self.train_en, self.train_cn, self.test_en, self.test_cn = self.split_data(self.en_encode, self.cn_encode,train_ratio)
        # 划分批次
        self.train_data = self.split_batch(self.train_en, self.train_cn,batch_size)
        self.test_data = self.split_batch(self.test_en, self.test_cn,batch_size)

    def load_data(self, path):
        """
        读取句子，并在句子后面加上停止符 <BOS> <EOS>
        """
        en = []
        cn = []
        with open(path, encoding = 'utf-8') as f:
            lines = f.read().strip().split('\n')
            for l in lines:
                en_sent, cn_sent, n = l.split('\t')
                # 英文
                en_sent = ["<BOS>"] + tokenize_en(en_sent) + ["<EOS>"]
                # 中文
                cn_sent = cht_to_chs(cn_sent) # 繁体 -> 简体
                cn_sent = ["<BOS>"] + tokenize_cn(cn_sent) + ["<EOS>"]
                en.append(en_sent)
                cn.append(cn_sent)
        return en, cn

    def build_vocab(self, sents):
        word_conts = Counter([word for sent in sents for word in sent])
        sorted_word = sorted(word_conts.items(), key = lambda  x : x[1], reverse=True)
        vocab = {word: idx+2 for idx, (word, count) in enumerate(sorted_word)}
        vocab['<PAD>'] = PAD
        vocab['<UNK>'] = UNK
        idx2word = { v:k for k,v in vocab.items()}
        return vocab, idx2word

    def encode_sentence(self, en_sents, cn_sents, en_vocab, cn_vocab, sort = True):
        """
        将英文、中文单词转化为单词索引列表
        """
        length = len(en_sents)
        en_encode = [[en_vocab.get(word, UNK) for word in sent] for sent in en_sents]
        cn_encode = [[cn_vocab.get(word, UNK) for word in sent] for sent in cn_sents]

        def len_argsort(seq):
            return sorted(range(len(seq)), key = lambda x : len(seq[x]))
        if sort:
            sorted_idx = len_argsort(en_encode)
            en_encode = [en_encode[idx] for idx in sorted_idx]
            cn_encode = [cn_encode[idx] for idx in sorted_idx]
        return en_encode, cn_encode

    def split_data(self, en_data, cn_data, train_ratio=0.8, shuffle=True):
        data_size = len(en_data)
        indices = list(range(data_size))
        if shuffle:
            random.shuffle(indices)

        train_end = int(data_size * train_ratio)

        train_en = [en_data[i] for i in indices[:train_end]]
        train_cn = [cn_data[i] for i in indices[:train_end]]

        test_en = [en_data[i] for i in indices[train_end:]]
        test_cn = [cn_data[i] for i in indices[train_end:]]

        return train_en, train_cn, test_en, test_cn

    def split_batch(self, en, cn, batch_size, shuffle = True):
        idx_list = np.arange(0, len(en), batch_size)
        # 起始索引随机打乱
        if shuffle:
            np.random.shuffle(idx_list)
        # 存放所有批次的语句索引
        batch_indexs = []
        for idx in idx_list:
            batch_indexs.append(np.arange(idx,min(idx + batch_size, len(en))))
        # 构建批次列表
        en_to_cn_batches = []
        cn_to_en_batches = []
        batches = []
        for batch_index in batch_indexs:
            # 按当前批次的样本索引采样
            batch_en = [en[index] for index in batch_index]
            batch_cn = [cn[index] for index in batch_index]
            # 对当前批次中所有语句填充、长度对齐
            batch_en = seq_padding(batch_en)
            batch_cn = seq_padding(batch_cn)
            # Batch类用于实现注意力掩码
            batches.append((Batch(batch_en, batch_cn), Batch(batch_cn, batch_en)))
            # en_to_cn_batches.append(Batch(batch_en, batch_cn))
            # cn_to_en_batches.append(Batch(batch_cn, batch_en))
        return batches

def subsequent_mask(size):
    "Mask out subsequent positions."
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)
    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return subsequent_mask == 0

class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src, tgt=None, pad=PAD):
        src = torch.from_numpy(src).long().to(device)
        tgt = torch.from_numpy(tgt).long().to(device)
        self.src = src
        # 对于当前输入的语句非空部分进行判断，bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            # 解码器使用的目标输入部分
            self.tgt = tgt[:, :-1]
            # 解码器训练时应预测输出的目标结果
            self.tgt_y = tgt[:, 1:]
            # 将目标输入部分进行注意力掩码
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

"""模型
    类：
        Embeddings              : 嵌入层
        PositionalEncoding      : 位置编码
        attention               : 注意力
        clones                  : 克隆基本单元，克隆的单元之间参数不共享
        MultiHeadedAttention    : 多头注意力机制
        LayerNorm               : 层归一化
        SublayerConnection      : 通过层归一化和残差连接，连接Multi-Head Attention和Feed Forward
        PositionwiseFeedForward : d层包括两层全连接层以及一个非线性激活函数ReLu
        EncoderLayer            : 编码单元
        Encoder                 : 编码器
        TransformerEncoder      : 生成编码器
        DecoderLayer            : 解码单元
        Decoder                 : 解码器
        TransformerDecoder      : 生成解码器
        Generator               : 解码器输出处理
        Transformer             : 模型主体
        BidirectionalTransformer: 双向翻译
        LabelSmoothing          : 标签平滑
        NoamOpt                 : 动态学习率
        BidirectionalTranslator : 双向翻译器
"""

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        # Embedding维数
        self.d_model = d_model

    def forward(self, x):
        # 返回x的词向量
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 位置编码矩阵，维度[max_len, embedding_dim]
        pe = torch.zeros(max_len, d_model, device = device)
        # 单词位置
        position = torch.arange(0.0, max_len, device = device)
        position.unsqueeze_(1)
        # 使用exp和log实现幂运算
        div_term = torch.exp(torch.arange(0.0, d_model, 2, device = device) * (- math.log(1e4) / d_model))
        div_term.unsqueeze_(0)
        # 计算单词位置沿词向量维度的纹理值
        pe[:, 0 : : 2] = torch.sin(torch.mm(position, div_term))
        pe[:, 1 : : 2] = torch.cos(torch.mm(position, div_term))
        # 增加批次维度，[1, max_len, embedding_dim]
        pe = pe.unsqueeze_(0)
        # 将位置编码矩阵注册为buffer(不参加训练)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将一个批次中语句所有词向量与位置编码相加
        # 注意，位置编码不参与训练，因此设置requires_grad=False
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

def attention(query, key, value, mask=None, dropout=None):
    """
    Scaled Dot-Product Attention
    """
    # q、k、v向量长度为d_k
    d_k = query.size(-1)
    # 矩阵乘法实现q、k点积注意力，sqrt(d_k)归一化
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 注意力掩码机制
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 注意力矩阵softmax归一化
    p_attn = scores.softmax(dim=-1)
    # dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 注意力对v加权
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    """
    克隆基本单元，克隆的单元之间参数不共享
    """
    return nn.ModuleList([
        copy.deepcopy(module) for _ in range(N)
    ])


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        """
        `h`：注意力头的数量
        `d_model`：词向量维数
        """
        # 确保整除
        assert d_model % h == 0
        # q、k、v向量维数
        self.d_k = d_model // h
        # 头的数量
        self.h = h
        # WQ、WK、WV矩阵及多头注意力拼接变换矩阵WO
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        # 批次大小
        nbatches = query.size(0)
        # WQ、WK、WV分别对词向量线性变换，并将结果拆成h块
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        # 注意力加权
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 多头注意力加权拼接
        x = (x.transpose(1, 2)
             .contiguous() # 确保内存连续
             .view(nbatches, -1, self.h * self.d_k))
        del query
        del key
        del value
        # 对多头注意力加权拼接结果线性变换
        return self.linears[-1](x)

class LayerNorm(nn.Module):
    """
    层归一化
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))   # 可学习的缩放参数
        self.b_2 = nn.Parameter(torch.zeros(features))  # 可学习的便宜参数
        self.eps = eps # 数值稳定性的小常数

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # 计算最后一个维度的均值
        std = x.std(-1, keepdim=True) # 计算最后一个维度的标准差
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    通过层归一化和残差连接，连接Multi-Head Attention和Feed Forward
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 层归一化
        x_ = self.norm(x)
        x_ = sublayer(x_)
        x_ = self.dropout(x_)
        # 残差连接
        return x + x_

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # SublayerConnection作用连接multi和ffn
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # d_model
        self.size = size

    def forward(self, x, mask):
        # 将embedding层进行Multi head Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # attn的结果直接作为下一层输入
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        """
        layer = EncoderLayer
        """
        super(Encoder, self).__init__()
        # 复制N个编码器基本单元
        self.layers = clones(layer, N)
        # 层归一化
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        循环编码器基本单元N次
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class TransformerEncoder(nn.Module):
    def __init__(self, N, d_model, h, d_ff, dropout):
        super(TransformerEncoder, self).__init__()
        self.attn = MultiHeadedAttention(h, d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.encoderLayer = EncoderLayer(d_model, self.attn, self.ff, dropout)
        self.encoder = Encoder(self.encoderLayer, N)
    def forward(self, x, mask):
        return self.encoder(x, mask)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        # 自注意力机制
        self.self_attn = self_attn
        # 上下文注意力机制
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # memory为编码器输出隐表示
        m = memory
        # 自注意力机制，q、k、v均来自解码器隐表示
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 上下文注意力机制：q为来自解码器隐表示，而k、v为编码器隐表示
        x = self.sublayer[1](x, lambda x: self.self_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        循环解码器基本单元N次
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class TransformerDecoder(nn.Module):
    def __init__(self, N, d_model, h, d_ff, dropout):
        super(TransformerDecoder, self).__init__()
        self.attn1 = MultiHeadedAttention(h, d_model)
        self.attn2 = MultiHeadedAttention(h, d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.decoderLayer = DecoderLayer(d_model, self.attn1, self.attn2, self.ff, dropout)
        self.decoder = Decoder(self.decoderLayer, N)
    def forward(self, x, memory, src_mask, tgt_mask):
        return self.decoder(x, memory,src_mask, tgt_mask)

class Generator(nn.Module):
    """
    解码器输出经线性变换和softmax函数映射为下一时刻预测单词的概率分布
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
        return log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self,  N, d_model, h, d_ff, dropout):
            super(Transformer, self).__init__()
            self.encoder = TransformerEncoder(N, d_model, h, d_ff, dropout)
            self.decoder = TransformerDecoder(N, d_model, h, d_ff, dropout)

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(tgt, memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # encoder的结果作为decoder的memory参数传入，进行decode
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

class BidirectionalTransformer(nn.Module):
    """
    双向翻译
    """
    def __init__(self, en_vocab_size, cn_vocab_size, N=6, d_model=512, d_ff = 2048, h = 8, dropout = 0.1):
        super(BidirectionalTransformer, self).__init__()
        # 分别定义中英文的嵌入层
        self.en_embedding = Embeddings(d_model, en_vocab_size)
        self.cn_embedding = Embeddings(d_model, cn_vocab_size)
        # 定义位置编码
        self.position = PositionalEncoding(d_model, dropout)
        # 共享的Transformer核心
        self.transformer = Transformer(N, d_model, h, d_ff, dropout)
        # 独立输出层
        self.en_generator = Generator(d_model, en_vocab_size)
        self.cn_generator = Generator(d_model, cn_vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_lang, src_mask=None):
        if src_lang == 'en':
            src_embed = self.en_embedding(src)
        else:
            src_embed = self.cn_embedding(src)

        src_embed = self.position(src_embed)
        return self.transformer.encoder(src_embed, src_mask)

    def decode(self,  tgt_lang, memory, src_mask, tgt, tgt_mask=None):
        if tgt_lang == 'en':
            tgt_embed = self.en_embedding(tgt)
        else:
            tgt_embed = self.cn_embedding(tgt)

        tgt_embed = self.position(tgt_embed)
        return self.transformer.decoder(tgt_embed, memory, src_mask,  tgt_mask)

    def forward(self, src, tgt, src_lang = 'en', tgt_lang = 'cn', src_mask = None, tgt_mask = None):
        # 确定方向
        # 源
        if src_lang == 'en':
            src_embed = self.en_embedding(src)
        else:
            src_embed = self.cn_embedding(src)
        # 目标
        if tgt_lang == 'en':
            tgt_embed = self.en_embedding(tgt)
            generator =  self.en_generator
        else:
            tgt_embed = self.cn_embedding(tgt)
            generator =  self.cn_generator
        # 添加位置编码
        src_embed = self.position(src_embed)
        tgt_embed = self.position(tgt_embed)
        # Transformer 处理
        output = self.transformer(src_embed, tgt_embed, src_mask = src_mask, tgt_mask = tgt_mask)
        return generator(output)


class LabelSmoothing(nn.Module):
    """
    标签平滑
    """
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


class BidirectionalTranslator:
    def __init__(self, model, en_tokenizer, cn_tokenizer, en_vocab, cn_vocab, device):
        self.model = model.to(device)
        self.en_tokenizer = en_tokenizer
        self.cn_tokenizer = cn_tokenizer
        self.en_vocab = en_vocab
        self.cn_vocab = cn_vocab
        self.device = device

        self.en_to_idx = {v: k for k, v in en_vocab.items()}
        self.cn_to_idx = {v: k for k, v in cn_vocab.items()}

    def translate(self, text, src_lang='en', tgt_lang='cn', max_len=50):
        self.model.eval()
        # 分词和编码
        if src_lang == 'en':
            tokens = ["<BOS>"] + self.en_tokenizer(text) + ["<EOS>"]
            tokens = [self.en_vocab.get(word, UNK) for word in tokens]
            vocab = self.cn_vocab
            idx_to_word = self.cn_to_idx
        else:
            tokens = ["<BOS>"] + self.cn_tokenizer(text) + ["<EOS>"]
            tokens = [self.cn_vocab.get(word, UNK) for word in tokens]
            vocab = self.en_vocab
            idx_to_word = self.en_to_idx

        src = torch.from_numpy(np.array(tokens)).long().to(device)
        # 增加一维
        src = src.unsqueeze(0)
        # 设置attention mask
        src_mask = (src != 0).unsqueeze(-2)
        # 用训练好的模型进行decode预测
        out = self.greedy_decode(self.model, src, src_mask,src_lang, tgt_lang, max_len=max_len, start_symbol=vocab["<BOS>"])
        # 初始化一个用于存放模型翻译结果语句单词的列表
        translation = []
        # 遍历翻译输出字符的下标（注意：开始符"BOS"的索引0不遍历）
        for j in range(1, out.size(1)):
            # 获取当前下标的输出字符
            sym = idx_to_word[out[0, j].item()]
            # 如果输出字符不为'EOS'终止符，则添加到当前语句的翻译结果列表
            if sym != '<EOS>':
                translation.append(sym)
            # 否则终止遍历
            else:
                break

        # 转换为文本
        if tgt_lang == 'en':
            return ' '.join(translation)
        else:
            return ''.join(translation)

    def en2cn(self, text):
        return self.translate(text, 'en', 'cn')

    def cn2en(self, text):
        return self.translate(text, 'cn', 'en')

    def greedy_decode(self, model, src, src_mask, src_lang, tgt_lang,max_len, start_symbol):
        """
        传入一个训练好的模型，对指定数据进行预测
        """
        # 先用encoder进行encode
        memory = model.encode(src, src_lang, src_mask)
        if tgt_lang == 'en':
            generator = model.en_generator
        else:
            generator = model.cn_generator
        # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
        # 遍历输出的长度下标
        for i in range(max_len - 1):
            # decode得到隐层表示
            out = model.decode(
                                tgt_lang,
                                memory,
                                src_mask,
                                ys,
                                subsequent_mask(ys.size(1)).type_as(src.data))
            # 将隐藏表示转为对词典各词的log_softmax概率分布表示
            prob = generator(out[:, -1])
            # 获取当前位置最大概率的预测词id
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            # 将当前位置预测的字符id与之前的预测内容拼接起来
            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        return ys


"""模型训练

"""


def train_epoch(data, model, cn_criterion,en_criterion, optimizer, epoch):
    total_tokens = 0.
    total_loss = 0.

    loop = tqdm(enumerate(data), total=len(data))

    model.train()
    for i, (en_to_cn_batches, cn_to_en_batches) in loop:
        optimizer.optimizer.zero_grad()
        # 英→中
        batch = en_to_cn_batches
        output = model(batch.src, batch.tgt, src_lang = 'en', tgt_lang = 'cn', src_mask = batch.src_mask, tgt_mask = batch.tgt_mask)
        loss = cn_criterion(output.contiguous().view(-1, output.size(-1)), batch.tgt_y.contiguous().view(-1))
        loss.backward()
        total_loss += loss
        total_tokens += batch.ntokens
        # 中→英
        batch = cn_to_en_batches
        output = model(batch.src, batch.tgt, src_lang = 'cn', tgt_lang = 'en', src_mask = batch.src_mask, tgt_mask = batch.tgt_mask)
        loss = en_criterion(output.contiguous().view(-1, output.size(-1)), batch.tgt_y.contiguous().view(-1))
        loss.backward()
        total_loss += loss
        total_tokens += batch.ntokens

        # 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        loop.set_description(f"Train Epoch [{epoch}/{epochs}")
        loop.set_postfix(loss = (total_loss / total_tokens).item())
    return total_loss / total_tokens

def test_epoch(data, model, cn_criterion,en_criterion, epoch):
    total_tokens = 0.
    total_loss = 0.

    loop = tqdm(enumerate(data), total=len(data))

    model.eval()
    with torch.no_grad():
        for i, (en_to_cn_batches, cn_to_en_batches) in loop:
            # 英→中
            batch = en_to_cn_batches
            output = model(batch.src, batch.tgt, src_lang = 'en', tgt_lang = 'cn', src_mask = batch.src_mask, tgt_mask = batch.tgt_mask)
            loss = cn_criterion(output.contiguous().view(-1, output.size(-1)),batch.tgt_y.contiguous().view(-1))
            total_loss += loss
            total_tokens += batch.ntokens

            # 中→英
            batch = cn_to_en_batches
            output = model(batch.src, batch.tgt, src_lang = 'cn', tgt_lang = 'en', src_mask = batch.src_mask, tgt_mask = batch.tgt_mask)
            loss = en_criterion(output.contiguous().view(-1, output.size(-1)),batch.tgt_y.contiguous().view(-1))
            total_loss += loss
            total_tokens += batch.ntokens

            loop.set_description(f"Test Epoch [{epoch}/{epochs}")
            loop.set_postfix(loss = (total_loss / total_tokens).item())
        return total_loss / total_tokens

def train(data, model, model_name, cn_criterion,en_criterion,  optimizer):
    """
    训练并保存数据
    """
    best_dev_loss = 1e5
    model.to(device)
    loss_s = []
    for epoch in range(epochs):

        train_loss = train_epoch(data.train_data, model,  cn_criterion,en_criterion,  optimizer, epoch)

        test_loss = test_epoch(data.test_data,model, cn_criterion,en_criterion,  epoch)
        loss_s.append(test_loss.item())
        print("Epoch {} Train-loss: {:.5f} Test-loss: {:.5f}".format
              (epoch, train_loss, test_loss))
        if test_loss < best_dev_loss:
            torch.save(model.state_dict(), 'save/{}_model.pt'.format(model_name))
            best_dev_loss = test_loss
    translator = BidirectionalTranslator(
        model,
        tokenize_en, tokenize_cn,
        data.en_vocab, data.cn_vocab,
        device
    )
    bleu = calculate_bleu(model, data, translator)
    return best_dev_loss, bleu, loss_s


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    传入一个训练好的模型，对指定数据进行预测
    """
    # 先用encoder进行encode
    memory = model.encode(src, src_mask)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 遍历输出的长度下标
    for i in range(max_len - 1):
        # decode得到隐层表示
        out = model.decode(memory,
                           src_mask,
                           ys,
                          subsequent_mask(ys.size(1)).type_as(src.data))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys



def calculate_bleu(model, data,translator):
    model.eval()
    en_cn_translations = []
    en_cn_references = []
    cn_en_translations = []
    cn_en_references = []

    with torch.no_grad():
        loop = tqdm(range(len(data.test_en)), total = len(data.test_en))
        for i in loop:
            en_sent = [data.en_idx2word[w] for w in data.test_en[i]]
            en_sent =' '.join(en_sent[1:-1])
            cn_sent = [data.cn_idx2word[w] for w in data.test_cn[i]]
            cn_sent = ''.join(cn_sent[1:-1])

            en_cn_translations.append(translator.en2cn(en_sent))
            en_cn_references.append(cn_sent)

            cn_en_translations.append(translator.cn2en(cn_sent))
            cn_en_references.append(en_sent)

            loop.set_description(f"BLEU")
    # 计算BLEU分数
    bleu = corpus_bleu(en_cn_translations, en_cn_references).score + corpus_bleu(cn_en_translations, cn_en_references).score
    return bleu / 2


# 定义不同的超参数配置
configurations = [
    {
        'name': 'base',
        'layers': 4,
        'd_model': 256,
        'h_num': 8,
        'd_ff': 1024,
        'dropout': 0.1,
    },
    {
        'name': 'small',
        'layers': 2,
        'd_model': 512,
        'h_num': 16,
        'd_ff': 2048,
        'dropout': 0.1,
    },
    {
        'name': 'deep',
        'layers': 4,
        'd_model': 1024,
        'h_num': 16,
        'd_ff': 2048,
        'dropout': 0.1,
    },
]

batch_size = 16
epochs = 20
max_length = 60
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main():
    data_path = 'cmn-eng/cmn.txt'
    data = PrepareData(data_path,0.8,batch_size)

    en_vocab_size = len(data.en_vocab)
    cn_vocab_size = len(data.cn_vocab)

    print("en_vocab %d" % en_vocab_size)
    print("cn_vocab %d" % cn_vocab_size)
    results = []
    df = pd.DataFrame(columns = ['model', 'epoch', 'loss'])
    for config in configurations:
        model = BidirectionalTransformer(
            en_vocab_size,
            cn_vocab_size,
            config['layers'],
            config['d_model'],
            config['d_ff'],
            config['h_num'],
            config['dropout']
        )
        cn_criterion = LabelSmoothing(cn_vocab_size, padding_idx=0, smoothing=0.0)
        en_criterion = LabelSmoothing(en_vocab_size, padding_idx=0, smoothing=0.0)
        optimizer = NoamOpt(
            config['d_model'],
            1,
            2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        loss, bleu, loss_s = train(data, model,config['name'],cn_criterion,en_criterion, optimizer)
        result = config
        result['loss'] = loss.item()
        result['BLEU'] = bleu
        results.append(result)
        for i in range(epochs):
            df.loc[len(df.index)] = [config['name'], i+1, loss_s[i]]
    results_df = pd.DataFrame(results)
    results_df.to_csv('save/result.csv', index=False)
    df.to_csv('save/loss.csv', index=False)


def pre():
    data_path = 'cmn-eng/cmn.txt'
    data = PrepareData(data_path,0.8,batch_size)

    en_vocab_size = len(data.en_vocab)
    cn_vocab_size = len(data.cn_vocab)

    config = configurations[1]
    model = BidirectionalTransformer(
        en_vocab_size,
        cn_vocab_size,
        config['layers'],
        config['d_model'],
        config['d_ff'],
        config['h_num'],
        config['dropout']
    )
    model.load_state_dict(torch.load('save/small_model.pt',weights_only=True))
    # 创建翻译器
    translator = BidirectionalTranslator(
        model,
        tokenize_en, tokenize_cn,
        data.en_vocab, data.cn_vocab,
        device
    )
    # 翻译
    print('原句: ',"Hello, how are you?")
    print('翻译: ', translator.en2cn("Hello, how are you?"))
    print('')
    print('原句: ',"你在干什么?")
    print('翻译: ', translator.cn2en("你在干什么?"))

if __name__ == '__main__':
    # main()
    pre()

