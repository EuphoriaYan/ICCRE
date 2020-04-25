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
from layers.bert_layernorm import BertLayerNorm


class BertTagger(nn.Module):
    def __init__(self, config, num_labels=4):
        super(BertTagger, self).__init__()
        self.num_labels = num_labels

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

        if config.classifier_sign == "single_linear":
            self.classifier = SingleLinearClassifier(config.hidden_size, self.num_labels) 
        elif config.classifier_sign == "multi_nonlinear":
            self.classifier = MultiNonLinearClassifier(config.hidden_size, self.num_labels)
        else:
            raise ValueError

        if config.use_crf:
            self.crf = CRF(num_labels, batch_first=True)

    # (input_ids, segment_ids, input_mask, label_ids)
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, char_mask=None, use_crf=False):
        last_bert_layer, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        last_bert_layer = last_bert_layer.view(-1, self.hidden_size)
        last_bert_layer = self.dropout(last_bert_layer)
        logits = self.classifier(last_bert_layer)
        # (seq_length, batch_size, num_tags)

        if labels is not None:
            if use_crf:
                loss = self.crf(logits.view(-1, input_ids.shape[1], self.num_labels), labels, attention_mask.byte())
                return -loss
            else:
                loss_fct = CrossEntropyLoss()
                if attention_mask is not None:
                    active_loss = (attention_mask.view(-1) == 1)
                    if char_mask is not None:
                        active_loss &= char_mask.view(-1)
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_label = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_label)
                    return loss
                else:
                    return loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            if use_crf:
                return self.crf.decode(logits.view(-1, input_ids.shape[1], self.num_labels), attention_mask.byte())
            else:
                logits = logits.detach().cpu().numpy()
                logits = np.reshape(logits, (-1, input_ids.shape[1], self.num_labels))
                logits = np.argmax(logits, axis=-1)
                logits = logits.tolist()
                return logits
