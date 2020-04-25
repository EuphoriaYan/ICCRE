#!/usr/bin/env python3 
# -*- coding: utf-8 -*-


import os
import sys
import csv
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm, trange


root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from utils.apply_text_norm import process_sent


class InputExample(object):
    # a single training / test example for simple sequence classification 
    def __init__(self, guid, text_a, text_b=None, label=None):
        """
        Construct s input Example. 
        Args:
            guid: unqiue id for the example.
            text_a: string, the untokenzied text of the first seq. for single sequence 
                tasks, only this sequction msut be specified. 
            text_b: (Optional) string, the untokenized text of the second sequence. 
            label: (Optional) string, the label of the example, This should be specific
                for train and dev examples, but not for test examples. 
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __str__(self):
        if self.text_b:
            return "guid:\t" + self.guid + "\n" + \
                   "text_a:\t" + self.text_a + "\n" + \
                   "text_b:\t" + self.text_b + "\n" + \
                   "label\t" + ' '.join(self.label) + "\n"
        else:
            return "guid:\t" + self.guid + "\n" + \
                   "text_a:\t" + self.text_a + "\n" + \
                   "label\t" + ' '.join(self.label) + "\n"


class InputFeature(object):
    # a single set of features of data
    def __init__(self, input_ids, input_mask, segment_ids, label_id, char_mask, label_len=0):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.char_mask = char_mask
        self.label_len = label_len


class DataProcessor(object):
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
        with open(input_file, "r", encoding='utf8', errors='ignore') as f:
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


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, task_sign="ner"):
    # load a data file into a list of "InputBatch"
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a, char_mask_a = tokenizer.tokenize(process_sent(example.text_a))
        tokens_b, char_mask_b = None, None
        if example.text_b:
            tokens_b, char_mask_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, char_mask_a, char_mask_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
                if char_mask_a is not None:
                    char_mask_a = char_mask_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        char_mask = [True] + char_mask_a + [False]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            if char_mask_b is not None:
                char_mask += char_mask_b + [False]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # input_mask = [1] * len(input_ids)
        # input_mask = [0] + [1] * (len(input_ids) - 2) + [0]
        input_mask = [1] * len(input_ids)

        # zero padding up to the sequence length
        padding = [0] * (max_seq_length - len(input_ids))
        char_padding = [False] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        char_mask += char_padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(char_mask) == max_seq_length

        if len(example.label) > max_seq_length - 2:
            example.label = example.label[: (max_seq_length - 2)]

        # Check the output should equal number of char.
        label_len = len(list(filter(None, char_mask)))
        if len(example.label) > label_len:
            example.label = example.label[: label_len]

        def convert_label_to_id(label, label_map, empty_ids, char_mask):
            idx = 0
            # [CLS] set for empty_ids
            label_ids = [label_map[empty_ids]]
            for i in range(1, len(char_mask)):
                if char_mask[i]:
                    label_ids.append(label_map[label[idx]])
                    idx += 1
                else:
                    label_ids.append(label_map[empty_ids])
            if len(label) != idx:
                print(label)
                print(char_mask)
                raise ValueError
            return label_ids

        if task_sign == "ner":
            try:
                label_id = convert_label_to_id(example.label, label_map, "O", char_mask)
            except ValueError:
                print(example.text_a)
        elif task_sign == "pos":
            label_id = convert_label_to_id(example.label, label_map, "B-o", char_mask)
        elif task_sign == "cws":
            label_id = convert_label_to_id(example.label, label_map, "S-SEG", char_mask)
        elif task_sign == "BIO_cws":
            label_id = convert_label_to_id(example.label, label_map, "B", char_mask)
        elif task_sign == "clf":
            label_id = label_map[example.label]
        elif task_sign == "cws+pos":
            label_id = convert_label_to_id(example.label, label_map, "B-o", char_mask)
        else:
            raise ValueError("task_sign not found: %s." % task_sign)

        features.append(InputFeature(input_ids=input_ids,
                                     input_mask=input_mask,
                                     segment_ids=segment_ids,
                                     label_id=label_id,
                                     char_mask=char_mask,
                                     label_len=label_len))

    return features


def convert_examples_to_gccre_features_bert(examples, component_dict, label_list, max_seq_length, tokenizer, task_sign="ner"):
    # load a data file into a list of "InputBatch"
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for example in examples:
        text = example.text_a
        example.text_b = ''
        for char in text:
            components = component_dict.get(char, [char])
            example.text_b += ''.join(components)

    for (ex_index, example) in tqdm(enumerate(examples)):
        tokens_a = tokenizer.tokenize(process_sent(example.text_a))
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair_first_a(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        # zero padding up to the sequence length
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if len(example.label) > max_seq_length - 2:
            example.label = example.label[: (max_seq_length - 2)]

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

        features.append(InputFeature(input_ids=input_ids,
                                     input_mask=input_mask,
                                     segment_ids=segment_ids,
                                     label_id=label_id))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, char_mask_a, char_mask_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            if char_mask_a is not None:
                char_mask_a.pop()
        else:
            tokens_b.pop()
            if char_mask_b is not None:
                char_mask_b.pop()


def _truncate_seq_pair_first_a(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_b) > 0:
            tokens_b.pop()
        else:
            tokens_a.pop()
