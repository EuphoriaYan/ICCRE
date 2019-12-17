#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, \
SequentialSampler

import csv 
import json
import logging 
import random 
import argparse  
import numpy as np
from tqdm import tqdm 

from dataset_readers.bert_data_utils import *


class LineCSSProcessor(DataProcessor):

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        # reads a tab separated value file.
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=" ", quotechar=quotechar)
            lines = []
            for line in reader:
                label_list = []
                for segment in line:
                    flag = True
                    for _ in segment:
                        if flag:
                            label_list.append('B')
                            flag = False
                        else:
                            label_list.append('I')
                sent = ''.join(line)
                assert len(sent) == len(label_list)
                lines.append([sent, label_list])
            return lines

    @classmethod
    def _read_lines(cls, input_lines):
        # reads white space separated lines.
        lines = []
        for line in input_lines:
            label_list = []
            line = line.strip().split(' ')
            for segment in line:
                flag = True
                for _ in segment:
                    if flag:
                        label_list.append('B')
                        flag = False
                    else:
                        label_list.append('I')
            sent = ''.join(line)
            assert len(sent) == len(label_list)
            lines.append([sent, label_list])
        return lines

    @classmethod
    def _read_lst_lines(cls, input_lines):
        # reads list separated lines.
        lines = []
        for line in input_lines:
            label_list = []
            for segment in line:
                flag = True
                for _ in segment:
                    if flag:
                        label_list.append('B')
                        flag = False
                    else:
                        label_list.append('I')
            sent = ''.join(line)
            assert len(sent) == len(label_list)
            lines.append([sent, label_list])
        return lines

    def get_train_examples(self, data_dir):
        # see base class
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_test_examples(self, data_dir):
        # see base class
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_dev_examples(self, data_dir):
        # see base class
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")

    @classmethod
    def get_labels(cls):
        return ['B', 'I', ]

    def _create_examples(self, lines, set_type):
        # create examples for the training and dev sets
        examples = []
        for (i, line) in enumerate(lines):
            if line == "\n":
                continue

            text_a = line[0]
            text_b = None
            label = line[1]
            guid = "{}_{}".format("ws.cws", str(i))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ShijiCSSProcessor(DataProcessor):

    @classmethod
    def _read_file(cls, input_file):
        # reads white space separated lines.
        lines = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    label_list = ['B'] + ['I'] * (len(line) - 1)
                    lines.append((line, label_list))
        return lines

    def get_train_examples(self, data_dir):
        lines = []
        for i in range(1, 8):
            lines += self._read_file(os.path.join(data_dir, "shi_text" + str(i) + ".txt"))
        return self._create_examples(lines, "train")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_file(os.path.join(data_dir, "pred.txt")), "test")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_file(os.path.join(data_dir, "shi_pred.txt")), "dev")

    @classmethod
    def get_labels(cls):
        return ['B', 'I', ]

    def _create_examples(self, lines, set_type):
        # create examples for the training and dev sets
        examples = []
        cur_line = ''
        cur_label = []
        for (i, line) in tqdm(enumerate(lines)):
            if len(cur_line) + len(line[0]) < 32:
                cur_line = cur_line + line[0]
                cur_label.extend(line[1])
            else:
                text_a = cur_line
                text_b = None
                label = cur_label
                cur_line = ''
                cur_label = []
                guid = "{}_{}".format("shiji.css", str(i))
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples