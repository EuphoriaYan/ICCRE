#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import random


root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from layers.mask_cross_entropy import MaskCrossEntropy
from utils.components import SubCharComponent
from utils.tokenization import BertTokenizer


class GccreEmbedding(nn.Module):
    """
    输入token_id，输出其对应的char embedding，gccre embedding或者两者的结合. config中的参数定义如下：
    dropout: float, dropout rate
    idx2char: dict, 单词到对应token_id的映射
    char_embsize: int, char embedding size
    gccre_embsize: int, gccre embedding size
    pretrained_char_embedding: numpy.ndarray 预训练字向量
    subchar_embsize: int, 部件模型的embedding_size
    use_batch_norm: bool, 是否使用batch normalization
    use_layer_norm: bool, 是否使用layer normalization
    fc_merge: bool, 是否将concat之后的向量过全连接
    output_size: bool, 输出向量的维度
    """

    def __init__(self, model_config, idx2char=None):
        super(GccreEmbedding, self).__init__()
        self.config = model_config

        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')

        with open(self.config.pretrained_char_embedding_path, encoding='utf-8') as char_emb:
            char_cnt, emb_size = char_emb.readline().strip().split()
            char_embbedding = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in char_emb)

        if idx2char is not None:
            self.config.idx2char = idx2char

        self.config.pretrained_char_embedding = np.zeros((len(self.config.idx2char), self.config.char_embsize), np.float)
        for i in range(len(idx2char)):
            c = idx2char[i]
            emb_c = char_embbedding.get(c)
            if emb_c is not None:
                self.config.pretrained_char_embedding[i] = emb_c

        self.char_embedding = nn.Embedding(len(self.config.idx2char), self.config.char_embsize)
        self.subchar_component = SubCharComponent(composing_func='GRU',
                                                  embedding_size=self.config.subchar_embsize,
                                                  hidden_size=self.config.gccre_embsize // 2,
                                                  config=self.config,
                                                  num_layers=1)

        self.drop = nn.Dropout(self.config.dropout)
        self.token_size = self.config.char_embsize + self.config.gccre_embsize

        if not self.config.fc_merge:
            assert self.token_size == self.config.output_size, \
                'No FC layer，Then token_size {} should equal to output_size {}'.format(self.token_size, self.config.output_size)

        if self.config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.token_size)
        if self.config.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.token_size)

        if self.config.fc_merge:
            self.fc_concat = nn.Linear(self.token_size, self.config.output_size)
        # 1280 * 21128
        self.gccre_classifier = nn.Linear(self.config.char_embsize + self.config.gccre_embsize, len(self.config.idx2char))
        self.gccre_classification_criterion = MaskCrossEntropy(self.config.loss_mask_ids)

        self.init_weights()

    def forward(self, data):  # 前向传播，输入输出加dropout，data:  (seq_len, batch)
        all_embeddings = []
        gccre_loss = []

        input_data = data.view(-1)
        # char_embedding (batch_size * max_seq_length * char_emb_size)
        all_embeddings.append(self.drop(self.char_embedding(input_data)))
        # subchar_embedding (batch_size * max_seq_length * subchar_emb_size)
        all_embeddings.append(self.subchar_component(input_data))
        emb = torch.cat(all_embeddings, -1)  # seql, batch, feat*2

        emb_logit = self.gccre_classifier(emb)
        emb_loss = self.gccre_classification_criterion(emb_logit, input_data)
        gccre_loss.append(emb_loss)

        if self.config.use_batch_norm:
            emb = self.batch_norm(emb)
        if self.config.use_layer_norm:
            emb = self.layer_norm(emb)

        if self.config.fc_merge:
            emb = F.relu(self.fc_concat(emb))

        gccre_classification_loss = sum(gccre_loss) / len(gccre_loss) if gccre_loss else 0
        out_shape = list(data.size())
        out_shape.append(self.config.output_size)
        return emb.view(*out_shape), gccre_classification_loss

    def init_weights(self):
        if self.config.char_embsize:
            # 有预训练好的词嵌入则载入，冻结训练
            if self.config.use_pretrained_char_embedding:
                self.char_embedding.weight = nn.Parameter(torch.FloatTensor(self.config.pretrained_char_embedding))
                for param in self.char_embedding.parameters():
                    param.requires_grad = False
            else:
                initrange = 0.1  # (-0.1, 0.1)的均匀分布，只对embedding和最后的线性层做初始化
                self.char_embedding.weight.data.uniform_(-initrange, initrange)
