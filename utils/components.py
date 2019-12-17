# encoding: utf-8
"""
使用text_rnn将每个token的部件模型压缩成一个向量
"""


import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
import torch.nn as nn
import collections

batch_size = 16

class SubCharComponent(nn.Module):

    def __init__(self, composing_func, embedding_size, hidden_size, config=None, num_layers=1):
        super(SubCharComponent, self).__init__()
        self.config = config
        self.composing_func = composing_func  # 构造函数：lstm/gru, cnn, avg, max
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.idx2char = self.config.idx2char
        self.char2idx = dict(zip(self.idx2char.values(), self.idx2char.keys()))

        if self.composing_func == 'LSTM':
            self.composing = nn.LSTM(input_size=embedding_size,
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     bidirectional=True)
        elif self.composing_func == 'GRU':
            self.composing = nn.GRU(input_size=embedding_size,  # 512
                                    hidden_size=hidden_size,  # 384
                                    num_layers=num_layers,
                                    bidirectional=True)

        component_dict_path = os.path.join(root_path, 'chaizi')
        self.component_dict = collections.OrderedDict()
        with open(os.path.join(component_dict_path, 'chaizi-jt.txt'), encoding='utf-8') as cz_jt:
            for line in cz_jt:
                item_list = line.strip().split('\t')
                self.component_dict[item_list[0]] = list(item_list[1].split(' '))

        self.embedding = nn.Embedding(len(self.idx2char), embedding_size)
        # 有预训练好的词嵌入则载入，冻结训练
        if self.config.use_pretrained_char_embedding:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(self.config.pretrained_char_embedding))
            for param in self.embedding.parameters():
                param.requires_grad = False
        else:
            initrange = 0.1  # (-0.1, 0.1)的均匀分布，只对embedding和最后的线性层做初始化
            self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, t_input):
        """
        根据字id，找到对应部件的id，过LSTM/GRU得到的字作为单词的表示
        :param t_input: (seq_len, batch_size)
        :return:
        """
        token_data = torch.Tensor([self.token_indexing(i, 'tokens') for i in t_input])
        token_len = torch.Tensor([self.token_indexing(i, 'lens') for i in t_input])
        tokens = t_input.new().long().new(*token_data.shape).copy_(token_data)
        token_lens = t_input.new().float().new(*token_len.shape).copy_(token_len)
        te = self.embedding(tokens)  # (batch_size, num_char, emb_size)
        reshaped_embeddings = te.permute(1, 0, 2)

        h0 = t_input.new().float().new(2, t_input.size()[0], self.hidden_size).zero_()
        to, _ = self.composing(reshaped_embeddings, h0)  # (seq_len, batch, num_directions * hidden_size)
        reshaped_outputs = to.permute(1, 0, 2)
        max_out, _ = torch.max(reshaped_outputs * token_lens.unsqueeze(-1), 1)  # (seq_len, batch, 2 * emb_size)
        return max_out

    def init_weight(self):
        # weight获得别的参数的数据类型和存储位置
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros((self.num_layers, batch_size, self.hidden_size)),
                    weight.new_zeros((self.num_layers, batch_size, self.hidden_size)))
        else:
            return weight.new_zeros((self.num_layers, batch_size, self.hidden_size))

    def token_indexing(self, idx, return_type):
        """
        将输入的字id映射为每个字部件的字符的id
        :param idx: (seq_len * batch_size)
        :return: chars: (seq_len, batch_size, num_char)  token_lens: (seq_len, batch_size, num_char)
        """
        c = self.idx2char[int(idx.cpu().numpy())]
        encoding = self.component_dict.get(c, [c])
        if len(encoding) > 8:
            encoding = encoding[:8]
        full_encoding = encoding if len(encoding) == 8 else encoding + ['[PAD]'] * (8 - len(encoding))
        assert len(full_encoding) == 8, full_encoding
        tokens = [self.char2idx.get(c, 100) for c in full_encoding]
        length = [i < len(encoding) for i in range(len(tokens))]
        # print(idx, c, encoding, tokens, length)

        if return_type == 'tokens':
            return tokens
        else:
            return length
