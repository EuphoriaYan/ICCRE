#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys


root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
import torch.nn as nn
from layers.crf import CRF


class BiLSTM_Clf(nn.Module):

    def __init__(self, config, num_labels):
        super().__init__()
        # Super Variable
        self.vocab_size = config.bert_config.vocab_size
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.lstm_dropout = config.lstm_dropout
        self.num_labels = num_labels
        self.num_layers = config.num_layers
        self.seq_length = config.max_seq_length

        # Layers
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.bilstm = nn.LSTM(self.embedding_size, self.hidden_size // 2,
                              num_layers=self.num_layers,
                              batch_first=True,
                              dropout=self.lstm_dropout,
                              bidirectional=True)
        self.pooling = nn.AvgPool1d(self.seq_length)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, char_mask=None, use_crf=True):
        embedding_features = self.embedding(input_ids)
        # Use batch first, so the input/output shape of lstm is (batch, seq, input_size/hidden_size)
        lstm_features = self.bilstm(embedding_features)[0]
        lstm_features = lstm_features.permute(0, 2, 1)
        # lstm_features = lstm_features.reshape(-1, self.hidden_size)
        pool_features = self.pooling(lstm_features)
        pool_features = pool_features.squeeze_()
        logits = self.classifier(pool_features)
        logits = logits.view(-1, self.num_labels)
        if labels is not None:
            return nn.CrossEntropyLoss(logits, labels)
        else:
            return self.softmax(logits)
