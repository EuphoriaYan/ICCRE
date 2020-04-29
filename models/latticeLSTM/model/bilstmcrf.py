# -*- coding: utf-8 -*-


import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-5])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch.nn as nn

from models.latticeLSTM.model.bilstm import BiLSTM
from models.latticeLSTM.model.crf import CRF


class BiLSTMCRF(nn.Module):
    def __init__(self, config, num_labels=4):
        super(BiLSTMCRF, self).__init__()
        self.num_labels = num_labels

        #  add two more label for downlayer lstm, use original label size for CRF
        num_labels += 2
        self.lstm = BiLSTM(config, num_labels)

        self.crf = CRF(self.num_labels)
        self.iteration = 0

    def neg_log_likelihood_loss(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs,
                                char_seq_lengths, char_seq_recover, batch_label, mask):
        outs = self.lstm.get_output_score(gaz_list, word_inputs, biword_inputs, word_seq_lengths,
                                                      char_inputs, char_seq_lengths, char_seq_recover)
        total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        scores, tag_seq = self.crf.viterbi_decode(outs, mask)

        return total_loss, tag_seq

    def forward(self, input_ids, input_mask, component_ids, component_len, label_ids):
        outs, gccre_loss = self.lstm.get_output_score(input_ids, input_mask, component_ids, component_len)
        scores, tag_seq = self.crf.viterbi_decode(outs, input_mask)
        return tag_seq

    def get_lstm_features(self, input_ids, input_mask, component_ids, component_len, label_ids):
        return self.lstm.get_lstm_features(input_ids, input_mask, component_ids, component_len, label_ids)
