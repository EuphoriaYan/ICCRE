# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-05-03 21:58:36

import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-5])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

device = "cuda:3"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from kblayer import GazLayer
from models.latticeLSTM.model.charbilstm import CharBiLSTM
from models.latticeLSTM.model.charcnn import CharCNN
from models.latticeLSTM.model.latticelstm import LatticeLSTM


class BiLSTM(nn.Module):
    def __init__(self, config, num_labels):
        super(BiLSTM, self).__init__()
        self.num_labels = num_labels

        self.use_gaz = config.use_gaz
        self.batch_size = config.batch_size

        self.embsize = config.embsize

        self.hidden_dim = config.hidden_dim

        self.dropout = nn.Dropout(config.dropout)
        self.lstm_dropout = nn.Dropout(config.dropout)

        self.embedding = nn.Embedding(config.vocab_size, self.embsize)

        self.bilstm_flag = config.use_bilstm

        # self.bilstm_flag = False
        self.lstm_layer = config.lstm_layer

        self.init_weight(config)

        # The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim.

        if self.bilstm_flag:
            lstm_hidden = config.hidden_dim // 2
        else:
            lstm_hidden = config.hidden_dim
        lstm_input = self.embsize

        self.forward_lstm = LatticeLSTM(input_dim=lstm_input,
                                        hidden_dim=lstm_hidden,
                                        word_drop=config.gaz_dropout,
                                        word_alphabet_size=config.vocab_size,
                                        word_emb_dim=config.embsize,
                                        pretrain_word_emb=config.pretrained_char_embedding,
                                        left2right=True,
                                        fix_word_emb=config.fix_gaz_emb)
        if self.bilstm_flag:
            self.backward_lstm = LatticeLSTM(input_dim=lstm_input,
                                             hidden_dim=lstm_hidden,
                                             word_drop=config.gaz_dropout,
                                             word_alphabet_size=config.vocab_size,
                                             word_emb_dim=config.embsize,
                                             pretrain_word_emb=config.pretrained_char_embedding,
                                             left2right=False,
                                             fix_word_emb=config.fix_gaz_emb)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(config.hidden_dim, self.num_labels)

    def init_weight(self, config):

        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')

        if config.gaz_file:
            with open(config.gaz_file, encoding='utf-8') as char_emb:
                char_cnt, emb_size = char_emb.readline().strip().split()
                char_embbedding = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in char_emb)
            config.pretrained_char_embedding = np.zeros((config.vocab_size, self.embsize), np.float)

            with open(os.path.join(config.bert_tokenizer, 'vocab.txt'), encoding='utf-8') as vocab_list:
                for i, vocab in enumerate(vocab_list):
                    emb_c = char_embbedding.get(vocab.strip())
                    if emb_c is not None:
                        config.pretrained_char_embedding[i] = emb_c

        if config.pretrained_char_embedding is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(config.pretrained_char_embedding))
        else:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(self.random_embedding(config.vocab_size, self.embsize)))

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def get_lstm_features(self, input_ids, input_mask, component_ids, component_len):
        """
            input:
                word_inputs: (batch_size, sent_len)
                gaz_list:
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output: 
                Variable(sent_len, batch_size, hidden_dim)
        """
        batch_size = component_ids.size(0)
        sent_len = component_ids.size(1)

        char_embs = self.embedding(input_ids)

        component_embs = self.embedding(component_ids)
        component_embs = self.dropout(component_embs)

        # packed_words = pack_padded_sequence(word_embs, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        skip_input = torch.LongTensor(np.zeros((batch_size, sent_len, 2))).to(device)
        for batch_i, (single_component_len, single_input_ids) in enumerate(zip(component_len, input_ids)):
            cur = 0
            for l, ids in zip(single_component_len, single_input_ids):
                if l == 0:
                    continue
                if cur + l >= sent_len:
                    break
                skip_input[batch_i][cur][0] = ids
                skip_input[batch_i][cur][1] = l
                cur += l

        lstm_out, hidden = self.forward_lstm(component_embs, skip_input, hidden)

        if self.bilstm_flag:
            backward_hidden = None
            backward_lstm_out, backward_hidden = self.backward_lstm(component_embs, skip_input, backward_hidden)
            lstm_out = torch.cat([lstm_out, backward_lstm_out], 2)

        lstm_out = self.lstm_dropout(lstm_out)
        return lstm_out

    def get_output_score(self, input_ids, input_mask, component_ids, component_len):
        lstm_out = self.get_lstm_features(input_ids, input_mask, component_ids, component_len)
        #  lstm_out (batch_size, sent_len, hidden_dim)
        outputs = self.hidden2tag(lstm_out)
        return outputs

    def neg_log_likelihood_loss(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs,
                                char_seq_lengths, char_seq_recover, batch_label, mask):
        #  mask is not used
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_word = batch_size * seq_len
        loss_function = nn.NLLLoss(ignore_index=0)
        outs = self.get_output_score(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs,
                                     char_seq_lengths, char_seq_recover)
        # outs (batch_size, seq_len, label_vocab)
        outs = outs.view(total_word, -1)
        score = F.log_softmax(outs, 1)
        loss = loss_function(score, batch_label.view(total_word))
        _, tag_seq = torch.max(score, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        return loss, tag_seq

    def forward(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                char_seq_recover, mask):

        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_word = batch_size * seq_len
        outs = self.get_output_score(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs,
                                     char_seq_lengths, char_seq_recover)
        outs = outs.view(total_word, -1)
        _, tag_seq = torch.max(outs, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        #  filter padded position with zero
        decode_seq = mask.long() * tag_seq
        return decode_seq
