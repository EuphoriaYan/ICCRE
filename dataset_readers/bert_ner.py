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


from bs4 import BeautifulSoup, Tag
import codecs


class ZztjNERProcessor(DataProcessor):

    def __init__(self):
        super(ZztjNERProcessor, self).__init__()

    # processor for the zztj dataset
    def _read_zztj_file(self, file_name):
        lines = []

        def analyse(line, tag):
            for child in tag.children:
                if type(child) == Tag and child['class'][0] == 'ano':
                    analyse(line, child)
                else:
                    line.append(str(child).strip())

        with open(file_name, 'r', encoding='utf-8') as f:
            raw_html = f.readline().strip("\"")
        html = raw_html.replace(r'\/', '/').replace(r'\"', '\"')
        # line = '<body>' + line + '</body>'
        soup = BeautifulSoup(html, 'lxml')
        soup = soup.p
        components = []
        analyse(components, soup)

        components = list(filter(None, components))
        processed_html = ''.join(components)
        processed_html = processed_html.replace('、', '').replace('，', '')
        processed_html = processed_html.replace('【', '').replace('】', '')

        split_pattern = re.compile(r'\\n|。|」|「|：|；|！|？|『|』')
        processed_html = re.split(split_pattern, processed_html)
        processed_html = [i.strip() for i in processed_html if i != '']

        for raw_line in processed_html:
            soup = BeautifulSoup(raw_line, 'lxml')
            soup = soup.body
            if soup is None:
                print(file_name)
                print(raw_line)
                continue
            if soup.p is not None:
                soup = soup.p
            line = ''
            label = []
            for child in soup.children:
                if type(child) == Tag:
                    cls = child['class'][0]
                    s = str(child.text)
                    line += s
                    label.extend([cls] * len(s))
                    if cls not in self.get_labels():
                        raise ValueError(cls)
                else:
                    s = str(child)
                    line += s
                    label.extend(['O'] * len(s))
            lines.append((line, label))
        return lines

    def get_train_examples(self, data_dir):
        lines = []
        for train_file in os.listdir(data_dir):
            num = int(train_file[:-4])
            if num < 250:
                lines.extend(self._read_zztj_file(os.path.join(data_dir, train_file)))
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir):
        lines = []
        for train_file in os.listdir(data_dir):
            num = int(train_file[:-3])
            if num >= 250:
                lines.extend(self._read_zztj_file(os.path.join(data_dir, train_file)))
        return self._create_examples(lines, "dev")

    def get_test_examples(self, data_dir):
        lines = []
        for train_file in os.listdir(data_dir):
            num = int(train_file[:-3])
            if num >= 250:
                lines.extend(self._read_zztj_file(os.path.join(data_dir, train_file)))
        return self._create_examples(lines, "test")

    def get_labels(self):
        return ['O', 'liter', 'peop', 'tpn', 'date', 'offi']

    def _create_examples(self, lines, set_type):
        # create examples for the training and dev sets
        examples = []
        for (i, line) in enumerate(lines):
            if line == "\n":
                continue
            text_a = line[0]
            text_b = None
            label = line[1]
            guid = "{}_{}".format("zztj.ner", str(i))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


if __name__ == '__main__':
    DataProcessor = ZztjNERProcessor()
    DataProcessor.get_train_examples('dataset/ner/zztj')
