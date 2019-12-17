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


class Ctb6CWSProcessor(DataProcessor):
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
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples 


class PkuCWSProcessor(DataProcessor):
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
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples 


class MsrCWSProcessor(DataProcessor):
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
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples 


class ZuozhuanCWSProcessor(DataProcessor):
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
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WhitespaceCWSPrecessor(DataProcessor):
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


class BookCWSProcessor(DataProcessor):

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

    def get_train_examples(self, data_dir):
        # see base class
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "tb.txt")), "train")

    def get_test_examples(self, data_dir):
        # see base class
        book_list = []
        book_name_list = []
        for maindir, subdir, file_name_list in os.walk(data_dir):
            book_name_list = file_name_list

        for book in book_name_list:
            print(book)
            path = os.path.join(data_dir, book)
            book_example = self._create_examples(self._read_tsv(path), "test")
            book_list.append(book_example)

        return book_list, book_name_list

    def get_dev_examples(self, data_dir):
        # see base class
        pass

    def get_labels(self):
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
            guid = "{}_{}".format("book.cws", str(i))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples