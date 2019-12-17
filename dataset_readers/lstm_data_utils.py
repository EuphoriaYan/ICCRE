#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import collections
import os
import sys



root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import csv
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from collections import OrderedDict
import torch

from utils.apply_text_norm import process_sent


class LSTMInputExample(object):
    # a single training / test example for simple sequence classification
    def __init__(self, guid, text_a, text_b=None, label=None):
        """
        Construct input Example.
        Args:
            guid: unqiue id for the example.
            text_a: string, the untokenzied text of the seq.
            label: (Optional) string, the label of the example, This should be specific
                for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class LSTMInputFeature(object):
    # a single set of features of data
    def __init__(self, input_ids, input_mask, component_ids, component_len, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.component_ids = component_ids
        self.component_len = component_len
        self.label_id = label_id


class LSTMDataProcessor(object):
    # base class for data converts for sequence classification datasets
    def get_train_examples(self, data_dir):
        # get a collection of "InputExample" for the train set
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        # gets a collections of "InputExample" for the dev set
        raise NotImplementedError()

    def get_labels(self):
        # gets the list of labels for this data set
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        # reads a tab separated value file.
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_csv(cls, input_file):
        # reads a tab separated value file.
        with open(input_file, "r", encoding='utf8', errors='ignore') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


def convert_examples_to_gccre_features_lstm(examples, component_dict, label_list, max_seq_length, tokenizer, task_sign="ner"):
    # load a data file into a list of "InputBatch"
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for example in tqdm(examples):
        text = example.text_a

        tokens = tokenizer.tokenize(process_sent(example.text_a))

        components = ''
        components_len = []
        for token in tokens:
            char_components = ''.join(component_dict.get(token, [token]))
            components_len.append(len(char_components))
            components += char_components

        components_tokens = tokenizer.tokenize(components)
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
        if len(components_tokens) > max_seq_length:
            components_tokens = components_tokens[max_seq_length]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        comp_ids = tokenizer.convert_tokens_to_ids(components_tokens)

        input_mask = [1] * len(comp_ids)

        # zero padding up to the sequence length
        comp_padding = [0] * (max_seq_length - len(comp_ids))
        comp_ids += comp_padding
        input_mask += comp_padding

        char_padding = [0] * (max_seq_length - len(input_ids))
        input_ids += char_padding
        components_len += char_padding

        assert len(comp_ids) == max_seq_length, text + '\n' + str(comp_ids)
        assert len(input_ids) == max_seq_length, text + '\n' + str(input_ids)
        assert len(components_len) == max_seq_length, text + '\n' + str(components_len)
        assert len(input_mask) == max_seq_length, text + '\n' + str(input_mask)

        if len(example.label) > max_seq_length:
            example.label = example.label[:max_seq_length]

        if task_sign == "ner":
            label_id = [label_map["O"]] + [label_map[tmp] for tmp in example.label] + [label_map["O"]]
            label_id += (len(input_ids) - len(label_id)) * [label_map["O"]]
        elif task_sign == "pos":
            label_id = [label_map["O"]] + [label_map[tmp] for tmp in example.label]
            label_id += (len(input_ids) - len(label_id)) * [label_map["O"]]
        elif task_sign == "cws":
            label_id = [label_map["S-SEG"]] + [label_map[tmp] for tmp in example.label]
            label_id += (len(input_ids) - len(label_id)) * [label_map["S-SEG"]]
        elif task_sign == "clf":
            label_id = label_map[example.label]
        else:
            raise ValueError

        features.append(LSTMInputFeature(input_ids=input_ids,
                                         input_mask=input_mask,
                                         component_ids=comp_ids,
                                         component_len=components_len,
                                         label_id=label_id))

    return features