#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-4])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch 
import torch.nn as nn 
from torch.nn import CrossEntropyLoss
import numpy as np


from layers.classifier import *
from layers.bert_basic_model import *
from layers.crf import CRF


class BertMultiTagger(nn.Module):
    def __init__(self, config, cws_labels=4, pos_labels=20):
        super(BertMultiTagger, self).__init__()
        self.cws_labels = cws_labels
        self.pos_labels = pos_labels

        bert_config = BertConfig.from_dict(config.bert_config.to_dict()) 
        self.bert = BertModel(bert_config)

        if config.bert_frozen == "true":
            print("!-!"*20) 
            print("Please notice that the bert grad is false")
            print("!-!"*20)
            for param in self.bert.parameters():
                param.requires_grad = False 

        self.hidden_size = config.hidden_size 
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier1 = nn.Linear(config.hidden_size, int(config.hidden_size/2))
        # self.classifier2 = nn.Linear(int(config.hidden_size/2), num_labels)

        # self.layer_norm = BertLayerNorm(config.hidden_size, )

        self.bert = self.bert.from_pretrained(config.bert_model, )

        self.cws_classifier = SingleLinearClassifier(config.hidden_size, self.cws_labels)
        self.cws_crf = CRF(self.cws_labels, batch_first=True)

        self.pos_classifier = SingleLinearClassifier(config.hidden_size, self.pos_labels)
        self.pos_crf = CRF(self.pos_labels, batch_first=True)

    # (input_ids, segment_ids, input_mask, label_ids)
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, cws_labels=None, pos_labels=None):
        last_bert_layer, pooled_output = self.bert(input_ids,
                                                   token_type_ids,
                                                   attention_mask,
                                                   output_all_encoded_layers=False)
        last_bert_layer = last_bert_layer.view(-1, self.hidden_size)
        last_bert_layer = self.dropout(last_bert_layer)
        cws_logits = self.cws_classifier(last_bert_layer)
        pos_logits = self.pos_classifier(last_bert_layer)
        # (seq_length, batch_size, num_tags)
        cws_logits = cws_logits.view(-1, input_ids.shape[1], self.cws_labels)
        pos_logits = pos_logits.view(-1, input_ids.shape[1], self.pos_labels)
        attention_mask = attention_mask.byte()
        if cws_labels is not None:
            cws_loss = -self.cws_crf(cws_logits, cws_labels, attention_mask)
            pos_loss = -self.pos_crf(pos_logits, pos_labels, attention_mask)
            return cws_loss + pos_loss
        else:
            cws_pred = self.cws_crf.decode(cws_logits, attention_mask)
            pos_pred = self.pos_crf.decode(pos_logits, attention_mask)
            return cws_pred, pos_pred
