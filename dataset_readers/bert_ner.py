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
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm
import re
from multiprocessing.pool import ThreadPool

from dataset_readers.bert_data_utils import *


class MsraNERProcessor(DataProcessor):
    # processor for the MSRA data set 
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.ner")), "train")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.ner")), "test")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.ner")), "dev")

    def get_labels(self):
        # see base class 
        return ["S-NS", "B-NS", "M-NS", "E-NS", "S-NR", "B-NR", "M-NR", "E-NR", \
                "S-NT", "B-NT", "M-NT", "E-NT", "O"]

    def _create_examples(self, lines, set_type):
        # create examples for the training and dev sets. 
        examples = []
        text_a = ''
        text_b = None
        label = []
        for (i, line) in enumerate(lines):
            if len(line) == 0:
                guid = "{}_{}".format("msra.ner", str(i))
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                text_a = ''
                label = []
                continue
            text_a += line[0]
            label.append(line[1])
            # guid = "{}_{}".format("msra.ner", str(i))
            # examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        # reads a tab separated value file.
        with open(input_file, "r", encoding='utf8', errors='ignore') as f:
            reader = csv.reader(f, delimiter=" ", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class OntoNotesNERProcessor(DataProcessor):
    # processor for OntoNotes dataset 
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_labels(self):
        # see base class 
        # {'M-GPE', 'S-LOC', 'M-PER', 'B-LOC', 'E-PER', 'M-LOC', 'B-PER', 'B-GPE', 
        # 'S-ORG', 'M-ORG', 'B-ORG', 'S-GPE', 'O', 'E-GPE', 'E-LOC', 'S-PER', 'E-ORG'}
        # GPE, LOC, PER, ORG, O
        return ["O", "S-LOC", "B-LOC", "M-LOC", "E-LOC", \
                "S-PER", "B-PER", "M-PER", "E-PER", \
                "S-GPE", "B-GPE", "M-GPE", "E-GPE", \
                "S-ORG", "B-ORG", "M-ORG", "E-ORG"]

    def _create_examples(self, lines, set_type):
        # create examples for the training and dev sets 
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            # continue 
            if line == "\n":
                continue

                # print("check the content of line")
            # print(line)
            # line.split("\t")
            text_a = line[0]
            text_b = None
            label = line[1]
            label = label.split(" ")
            guid = "{}_{}".format("ontonotes.ner", str(i))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ResumeNERProcessor(DataProcessor):
    # processor for the Resume dataset 
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_labels(self):
        return ["O", "S-ORG", "S-NAME", "S-RACE", "B-TITLE", "M-TITLE", "E-TITLE", "B-ORG", "M-ORG", "E-ORG", "B-EDU",
                "M-EDU", "E-EDU", "B-LOC", "M-LOC", "E-LOC", "B-PRO", "M-PRO", "E-PRO", "B-RACE", "M-RACE", "E-RACE",
                "B-CONT", "M-CONT", "E-CONT", "B-NAME", "M-NAME", "E-NAME", ]

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
            guid = "{}_{}".format("resume.ner", str(i))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


from bs4 import BeautifulSoup
import codecs


class ZztjNERProcessor(DataProcessor):

    def __init__(self):
        self.dev_list = ['史記 書-曆書第四.txt',
                         '史記 書-律書第三.txt',
                         '史記 書-平準書第八.txt',
                         '史記 書-天官書第五.txt',
                         '史記 書-樂書第二.txt']
        self.ner_tag = ['NB1', 'NB2', 'NB3', 'NB4', 'NB5']
        self.ner_word = set()

    # processor for the Sinica dataset
    def _read_zztj_file(self, file_name):
        lines = []

        with open(file_name, 'r', encoding='utf-8') as f:
            line = f.readline().strip("\"")
            line = line.replace(r'\/', '/')
            line = line.replace(r'\"', '\"')
            soup = BeautifulSoup(line, 'lxml')
            print(soup.text)
            # print(codecs.getdecoder("unicode_escape")(line)[0])

        # delete empty str
        lines = list(filter(None, lines))

        return

    def get_train_examples(self, data_dir):
        lines = []
        for train_file in os.listdir(data_dir):
            if "史記" in train_file and train_file not in self.dev_list:
                lines.extend(self._read_iis_file(os.path.join(data_dir, train_file)))
        print(self.ner_word)
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir):
        lines = []
        for dev_file in self.dev_list:
            lines.extend(self._read_iis_file(os.path.join(data_dir, dev_file)))
        return self._create_examples(lines, "dev")

    def get_test_examples(self, data_dir):
        lines = []
        for test_file in os.listdir(data_dir):
            if "左傳" in test_file:
                lines.extend(self._read_iis_file(os.path.join(data_dir, test_file)))
        return self._create_examples(lines, "test")

    def get_raw_labels(self):
        return ['O', 'A', 'C', 'DA', 'DB', 'DC', 'DD', 'DF', 'DG', 'DH', 'DJ', 'DL', 'DN', 'DV', 'I',
                'NA1', 'NA2', 'NA3', 'NA4', 'NA5', 'NB1', 'NB2', 'NB3', 'NB4', 'NB5',
                'NF', 'NG', 'NH', 'NI', 'P', 'S', 'T', 'U', 'VA', 'VC1', 'VC2',
                'VD', 'VE', 'VF', 'VG', 'VH1', 'VH2', 'VI', 'VJ', 'VK', 'VM', 'VP']

    def get_labels(self):
        return ['O', 'NB1', 'NB2', 'NB3', 'NB4', 'NB5']

    def _create_examples(self, lines, set_type):
        # create examples for the training and dev sets
        examples = []
        for (i, line) in enumerate(lines):
            if line == "\n":
                continue
            text_a = line[0]
            text_b = None
            label = line[1]
            guid = "{}_{}".format("sinica.ner", str(i))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


if __name__ == '__main__':
    DataProcessor = ZztjNERProcessor()
    DataProcessor._read_zztj_file('dataset/ner/zztj/1.txt')
