#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 

import os 
import sys 

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import json
import math
import copy
import logging
import tarfile
import tempfile
import shutil
import numpy as np


import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


from layers.bert_basic_model import *
from layers.gccre_position_embed import GccrePositionEmbedder


class GccreTransformer(nn.Module):
    def __init__(self, config, num_labels=4):
        super(GccreTransformer, self).__init__()
        self.num_labels = num_labels
        self.gccre_embedder = GccrePositionEmbedder(config.gccre_config)
        self.gccre_embedder = self.gccre_embedder.from_pretrained(config.gccre_config)
        bert_config = BertConfig.from_dict(config.bert_config.to_dict())
        self.bert_model = BertModel(bert_config)
        self.bert_model = self.bert_model.from_pretrained(config.gccre_config.bert_model)
        self.transformer_layer = BertEncoder(config.transformer_config)
        self.pooler = BertPooler(config)
        if config.bert_frozen == "true":
            print("!=!"*20)
            print("Please notice that the bert model if frozen")
            print("the loaded weights of models is ")
            print(config.gccre_config.bert_model)
            print("!-!"*20)
            for param in self.bert_model.parameters():
                param.requires_grad=False

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,):
        gccre_embed, gccre_cls_loss = self.gccre_embedder(input_ids, token_type_ids=token_type_ids)
        last_bert_layer, pooled_output = self.bert_model(input_ids,
                                                         token_type_ids,
                                                         attention_mask,
                                                         output_all_encoded_layers=False)

        input_features = torch.cat([gccre_embed, last_bert_layer], -1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * - 10000.0
        transformer_output = self.transformer_layer(input_features,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)
        sequence_output = transformer_output[-1]
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output, gccre_cls_loss
