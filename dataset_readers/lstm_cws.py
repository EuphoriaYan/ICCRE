#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys


root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


class SequentialSampler(object):
    pass


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import csv
import json
import logging
import random
import argparse
import numpy as np
from tqdm import tqdm

from dataset_readers.lstm_data_utils import *


class Ctb6CWSProcessor(LSTMDataProcessor):
    # processor for CTB6 CWS dataset
    def get_train_examples(self, data_dir):
        # see base class
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.char.bmes")), "train")


    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.char.bmes")), "test")


    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.char.bmes")), "dev")


    def get_labels(self):
        return ['E-SEG', 'S-SEG', 'B-SEG', 'M-SEG', ]


    def _create_examples(self, lines, set_type):
        # create examples for the training and dev sets
        examples = []
        for (i, line) in enumerate(lines):
            if line == "\n":
                continue

            text_a = line[0]
            text_b = None
            label = line[1]
            label = label.split(" ")
            guid = "{}_{}".format("ctb6.cws", str(i))
            examples.append(LSTMInputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class PkuCWSProcessor(LSTMDataProcessor):
    # processor for PKU CWS dataset
    def get_train_examples(self, data_dir):
        # see base class
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_labels(self):
        return ['B-SEG', 'M-SEG', 'S-SEG', 'E-SEG',]

    def _create_examples(self, lines, set_type):
        # create examples for the trainng and dev sets
        examples = []
        for (i, line) in enumerate(lines):
            if line == "\n":
                continue

            text_a = line[0]
            text_b = None
            label = line[1]
            label = label.split(" ")
            guid = "{}_{}".format("pku.cws", str(i))
            examples.append(LSTMInputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MsrCWSProcessor(LSTMDataProcessor):
    # processor for MSR CWS dataset
    def get_train_examples(self, data_dir):
        # see base class
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_test_examples(self, data_dir):
        # see base class
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_dev_examples(self, data_dir):
        # see base class
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_labels(self):
        return ['S-SEG', 'M-SEG', 'B-SEG', 'E-SEG',]

    def _create_examples(self, lines, set_type):
        # create examples for the training and dev sets
        examples = []
        for (i, line) in enumerate(lines):
            if line == "\n":
                continue
            text_a = line[0]
            text_b = None
            label = line[1]
            label = label.split(" ")
            guid = "{}_{}".format("mrs.cws", str(i))
            examples.append(LSTMInputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ZuozhuanCWSLSTMProcessor(LSTMDataProcessor):
    # processor for Personal classical CWS dataset

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        # reads a tab separated value file.
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="|", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def get_train_examples(self, data_dir):
        # see base class
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "tb.txt")), "train")

    def get_test_examples(self, data_dir):
        # see base class
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "ts.txt")), "test")

    def get_dev_examples(self, data_dir):
        # see base class
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "ts.txt")), "dev")

    def get_labels(self):
        return ['B', 'I', ]

    def _create_examples(self, lines, set_type):
        # create examples for the training and dev sets
        examples = []
        trans = {"[BOS]": "B",
                 "[IOS]": "I"}
        for (i, line) in enumerate(lines):
            if line == "\n":
                continue

            text_a = line[0]
            text_b = None
            label = line[1]
            label = label.split(" ")
            label = [trans[l] for l in label]
            guid = "{}_{}".format("per.cws", str(i))
            examples.append(LSTMInputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples