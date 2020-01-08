#!/usr/bin/env python3 
# -*- coding: utf-8 -*-


import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import csv 
import logging 
import argparse 
import random 
import numpy as np 
from tqdm import tqdm
import re

from dataset_readers.bert_data_utils import DataProcessor, InputExample
from utils.apply_text_norm import process_sent


class Ctb5POSProcessor(DataProcessor):
    # process for the Ctb5 pos processor 
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_test_examples(self, data_dir): 
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_labels(self):
        return ['O', 'B-FW', 'S-BA', 'B-PN', 'B-NR', 'B-M', 'M-NT', 'M-AD', 'E-P', 'M-CC', 'M-P', 'M-CD', 'S-CS',
                'M-NN-SHORT', 'B-MSP', 'S-CC', 'E-SP', 'E-NN', 'B-ETC', 'S-PN', 'B-NT', 'E-FW', 'S-NT-SHORT',
                'S-DER', 'B-PU', 'S-NT', 'B-AD', 'S-DT', 'E-VE', 'S-SP', 'E-IJ', 'M-CS', 'S-LB', 'B-NN',
                'S-VA', 'S-ETC', 'E-JJ', 'B-P', 'M-FW', 'B-LC', 'S-MSP', 'S-AS', 'S-NN', 'E-ETC', 'B-CC',
                'M-VA', 'E-ON', 'S-PU', 'E-DT', 'B-CS', 'S-IJ', 'E-PU', 'S-AD', 'S-M', 'E-LC', 'B-OD',
                'S-LC', 'M-PN', 'E-NR', 'E-M', 'M-NR', 'E-VC', 'B-NN-SHORT', 'E-NT', 'E-CD', 'S-NR', 'S-VV',
                'E-AD', 'B-JJ', 'B-DT', 'B-ON', 'M-DT', 'M-NN', 'S-SB', 'M-VV', 'S-DEG', 'S-ON', 'S-DEV',
                'S-NR-SHORT', 'E-CC', 'M-M', 'E-NN-SHORT', 'B-VV', 'S-P', 'S-JJ', 'E-VA', 'M-JJ', 'E-VV',
                'M-OD', 'B-VA', 'B-IJ', 'S-CD', 'E-CS', 'B-CD', 'B-VE', 'E-OD', 'S-OD', 'S-X', 'E-MSP',
                'S-FW', 'E-PN', 'B-VC', 'M-PU', 'M-VC', 'S-VC', 'S-DEC', 'S-VE', 'B-SP']


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
            guid = "{}_{}".format("ctb5pos", str(i)) 
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples 



class Ctb6POSProcessor(DataProcessor):
    # process for the ctb6 pos processor 
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.char.bmes")), "train")


    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.char.bmes")), "dev")


    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.char.bmes")), "test")


    def get_labels(self):
        # see base class 
        return ['O', 'E-NR', 'E-OD', 'B-SP', 'S-AS', 'M-P', 'M-JJ', 'E-IJ', 'S-OD', 'M-OD', 'M-VA', 'E-ETC', 'B-CC', 'S-MSP', 'B-LC', 'E-VA', 'E-VC', 'E-DT', 'M-VC', 'S-PN', 'E-MSP', 'M-PU', 'E-VE', 'B-DT', 'S-CC', 'S-DT', 'S-DER', 'B-AD', 'S-VV', 'S-NR', 'B-OD', 'S-VE', 'B-NN-SHORT', 'S-LB', 'S-CS', 'M-CC', 'E-PN', 'E-P', 'M-NN', 'S-DEC', 'E-PU', 'M-M', 'B-PU', 'M-PN', 'S-NN', 'B-M', 'M-DT', 'S-SB', 'B-CS', 'S-SP', 'M-CD', 'B-VE', 'S-ON', 'B-PN', 'B-P', 'S-VC', 'B-VA', 'S-FW', 'B-ON', 'S-NT-SHORT', 'E-NN-SHORT', 'M-VV', 'S-DEG', 'E-ON', 'S-NT', 'S-IJ', 'S-AD', 'M-FW', 'M-AD', 'B-CD', 'S-LC', 'E-CD', 'E-JJ', 'B-IJ', 'E-NN', 'E-SP', 'S-P', 'S-VA', 'S-ETC', 'B-VV', 'E-CS', 'S-CD', 'E-M', 'B-MSP', 'S-JJ', 'E-LC', 'S-PU', 'B-ETC', 'M-NT', 'E-CC', 'B-NN', 'S-BA', 'E-NT', 'E-AD', 'M-NR', 'B-NT', 'M-CS', 'B-JJ', 'S-M', 'S-X', 'S-DEV', 'S-NR-SHORT', 'B-NR', 'M-NN-SHORT', 'B-VC', 'E-FW', 'E-VV', 'B-FW']


    def _create_examples(self, lines, set_type):
        # create examples for the training and dev set 
        examples = []
        for (i, line) in enumerate(lines):
            if line == "\n":
                continue 

            text_a = line[0]
            text_b = None 
            label = line[1]
            laebl = label.split(" ")
            guid = "{}_{}".format("ctb6pos", str(i))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples 



class Ud1POSProcessor(DataProcessor):
    # process for the ud1 pos processor 
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.char.bmes")), "train")


    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.char.bmes")), "dev")


    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.char.bmes")), "test")


    def get_labels(self):
        return ['O', 'B-PART', 'M-ADV', 'B-CONJ', 'E-SYM', 'S-PROPN', 'S-PUNCT', 'B-ADP', 'S-PART', 'B-PUNCT', 'B-PRON', 'E-PRON', 'B-NOUN', 'E-ADP', 'E-NOUN', 'M-SYM', 'S-ADV', 'B-AUX', 'E-VERB', 'M-NUM', 'M-VERB', 'S-ADP', 'E-AUX', 'B-X', 'E-ADV', 'E-PROPN', 'S-AUX', 'M-X', 'S-VERB', 'B-PROPN', 'M-DET', 'M-PUNCT', 'E-PUNCT', 'S-DET', 'B-SYM', 'M-ADJ', 'S-NOUN', 'S-NUM', 'B-NUM', 'E-DET', 'B-VERB', 'S-CONJ', 'M-NOUN', 'S-SYM', 'E-NUM', 'B-ADJ', 'M-PART', 'S-PRON', 'E-ADJ', 'E-X', 'M-ADP', 'E-PART', 'M-PROPN', 'M-CONJ', 'S-X', 'B-ADV', 'S-ADJ', 'E-CONJ', 'B-DET']

    def _create_examples(self, lines, set_type):
        # create examples for the training and dev set 
        examples = []
        for (i, line) in enumerate(lines):
            if line == "\n":
                continue 

            text_a = line[0]
            text_b = None 
            label = line[1]
            label = label.split(" ")
            guid = "{}_{}".format("ud1pos", str(i))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples 


class ZuozhuanPOSProcessor(DataProcessor):
    # processor for the Resume dataset

    def __init__(self):
        self.tag_set = set()

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_file(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        lines = self._read_file(os.path.join(data_dir, "dev.txt"))
        # lines = lines[:16]
        return self._create_examples(lines, "dev")

    def get_test_examples(self, data_dir):
        lines = self._read_file(os.path.join(data_dir, "test.txt"))
        # lines = lines[:16]
        return self._create_examples(lines, "test")

    def get_labels(self):
        """
        return ['B-a', 'B-c', 'B-d', 'B-f', 'B-m', 'B-mr', 'B-n', 'B-nn', 'B-nr', 'B-ns', 'B-nsr',
                'B-p', 'B-r', 'B-rn', 'B-rr', 'B-rs', 'B-s', 'B-t', 'B-u', 'B-v', 'B-vn', 'B-y',
                'E-a', 'E-c', 'E-d', 'E-f', 'E-m', 'E-mr', 'E-n', 'E-nn', 'E-nr', 'E-ns', 'E-nsr',
                'E-p', 'E-r', 'E-rn', 'E-rr', 'E-rs', 'E-s', 'E-t', 'E-u', 'E-v', 'E-vn', 'E-y',
                'M-a', 'M-c', 'M-f', 'M-m', 'M-n', 'M-nr', 'M-ns', 'M-r', 'M-rr', 'M-t', 'M-v', 'M-y',
                'S-a', 'S-b', 'S-c', 'S-d', 'S-f', 'S-j', 'S-m', 'S-mr', 'S-n', 'S-nr', 'S-ns',
                'S-p', 'S-q', 'S-r', 'S-rs', 'S-s', 'S-sv', 'S-t', 'S-u', 'S-v', 'S-wv', 'S-y', 'S-yv',
                'S-o']  # S-o is for [CLS],[SEP],[PAD]
        """
        return ['B-a', 'B-b', 'B-c', 'B-d', 'B-f', 'B-j', 'B-m', 'B-mr', 'B-n', 'B-nn', 'B-nr', 'B-ns',
                'B-nsr', 'B-o', 'B-p', 'B-q', 'B-r', 'B-rn', 'B-rr', 'B-rs', 'B-s', 'B-sv', 'B-t', 'B-u',
                'B-v', 'B-vn', 'B-wv', 'B-y', 'B-yv', 'I-a', 'I-c', 'I-d', 'I-f', 'I-m', 'I-mr', 'I-n',
                'I-nn', 'I-nr', 'I-ns', 'I-nsr', 'I-p', 'I-r', 'I-rn', 'I-rr', 'I-rs', 'I-s', 'I-t',
                'I-u', 'I-v', 'I-vn', 'I-y'] # B-o is for [CLS],[SEP],[PAD]

    def get_word_cws_tag(self, word):
        assert len(word)>0
        if len(word) == 1:
            return ['B']
        else:
            return ['B'] + ['I'] * (len(word) - 1)

    def _read_file(self, file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()
        raw_lines = [raw_line.strip() for raw_line in raw_lines]
        lines = []
        for raw_line in raw_lines:
            if not raw_line:
                continue
            words = raw_line.split()
            sent = ''
            tag = []
            for word in words:
                word = word.split('/')
                w = word[0]
                sent += w
                ner_tag = word[1]
                cws_tag = self.get_word_cws_tag(w)
                tag.extend([ct + '-' + ner_tag for ct in cws_tag])
            for t in tag:
                self.tag_set.add(t)
            assert len(sent) == len(tag)
            lines.append((sent, tag))
        return lines

    def _create_examples(self, lines, set_type):
        # create examples for the training and dev sets
        examples = []
        for (i, line) in enumerate(lines):
            if line == "\n":
                continue
            text_a = line[0]
            text_b = None
            label = line[1]
            guid = "{}_{}".format("zz.ner", str(i))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ZuozhuanPosProcessor_multi_output(DataProcessor):
    # processor for the Resume dataset

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_file(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        lines = self._read_file(os.path.join(data_dir, "dev.txt"))
        # lines = lines[:16]
        return self._create_examples(lines, "dev")

    def get_test_examples(self, data_dir):
        lines = self._read_file(os.path.join(data_dir, "test.txt"))
        # lines = lines[:16]
        return self._create_examples(lines, "test")

    def get_cws_labels(self):
        return ['B', 'I', ]

    def get_pos_labels(self):
        return ['a', 'b', 'c', 'd', 'f', 'j', 'm', 'mr', 'n', 'nn', 'nr', 'ns', 'nsr', 'o', 'p', 'q',
                'r', 'rn', 'rr', 'rs', 's', 'sv', 't', 'u', 'v', 'vn', 'wv', 'y', 'yv', ]

    def get_word_cws_tag(self, word):
        assert len(word) > 0
        if len(word) == 1:
            return ['B']
        else:
            return ['B'] + ['I'] * (len(word) - 1)

    def _read_file(self, file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()
        raw_lines = [raw_line.strip() for raw_line in raw_lines]
        lines = []
        for raw_line in raw_lines:
            if not raw_line:
                continue
            words = raw_line.split()
            sent = ''
            pos_tags = []
            cws_tags = []

            for word in words:
                word = word.split('/')
                w = word[0]
                sent += w
                pos_tags.extend([word[1]] * len(w))
                cws_tags.extend(self.get_word_cws_tag(w))
            assert len(sent) == len(cws_tags) == len(pos_tags), \
                sent + '\n' + ','.join(cws_tags) + '\n' + ','.join(pos_tags)
            lines.append((sent, cws_tags, pos_tags))
        return lines

    def _create_examples(self, lines, set_type):
        # create examples for the training and dev sets
        examples = []
        for (i, line) in enumerate(lines):
            if line == "\n":
                continue
            text_a = line[0]
            text_b = None
            cws_label = line[1]
            pos_label = line[2]
            guid = "{}_{}".format("zz.ner", str(i))
            examples.append(InputExample2(guid=guid, text_a=text_a, text_b=text_b,
                                          label=cws_label, label2=pos_label))
        return examples


class InputExample2(object):
    # a single training / test example for simple sequence classification
    def __init__(self, guid, text_a, text_b=None, label=None, label2=None):
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
        self.label2 = label2

    def __str__(self):
        if self.text_b:
            return "guid:\t" + self.guid + "\n" + \
                   "text_a:\t" + self.text_a + "\n" + \
                   "text_b:\t" + self.text_b + "\n" + \
                   "label:\t" + ','.join(self.label) + "\n" + \
                   "label2:\t" + ','.join(self.label2) + "\n"
        else:
            return "guid:\t" + self.guid + "\n" + \
                   "text_a:\t" + self.text_a + "\n" + \
                   "label:\t" + ','.join(self.label) + "\n" + \
                   "label2:\t" + ','.join(self.label2) + "\n"


class InputFeature2(object):
    # a single set of features of data
    def __init__(self, input_ids, input_mask, segment_ids, label_id, label_id2=None, label_len=0):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.label_id2 = label_id2
        self.label_len = label_len


class Trie:

    def __init__(self):
        self.root = {}

    def insert(self, word: str):
        node = self.root
        for char in word:
            node = node.setdefault(char, {})

    def search(self, word: str):
        node = self.root
        res = ''
        for char in word:
            if char not in node:
                return res
            res += char
            node = node[char]
        return res

    def startsWith(self, prefix: str):
        node = self.root
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True


class SinicaPOSProcessor(DataProcessor):

    def __init__(self):
        self.dev_list = ['史記 書-曆書第四.txt',
                         '史記 書-律書第三.txt',
                         '史記 書-平準書第八.txt',
                         '史記 書-天官書第五.txt',
                         '史記 書-樂書第二.txt']
        self.ner_tag = ['NB1', 'NB2', 'NB3', 'NB4', 'NB5']
        self.ner_word = set()
        self.trie = Trie()
        for label in self.get_labels():
            self.trie.insert(label)

    # processor for the Sinica dataset
    def _read_iis_file(self, file_name):
        lines = []

        with open(file_name, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()
            for raw_line in raw_lines:
                sub_sent_lst = re.split('[。，！？：；「」『』、]', raw_line.strip())
                lines.extend(sub_sent_lst)
        # delete empty str
        lines = list(filter(None, lines))
        pattern = re.compile('(.*?)\((.*?)\)(\[.*?\])?', re.S)
        res_lines = []
        for line in lines:
            res_line = ''
            res_label = []
            tag = re.findall(pattern, line)
            for i, j, k in tag:
                if j.startswith('NA1') and k == '[+others]':
                    self.ner_word.add(i)
            tag = [(i, self.trie.search(j.upper())) for i, j, k in tag]
            tag = [(i, j if j in self.ner_tag else 'O') for i, j in tag]
            # tag = [self.trie.search(t.upper()) for t in tag]
            # tag = [t if t in self.ner_tag else 'O' for t in tag]
            for token, label in tag:
                res_line += token
                res_label.extend([label] * len(token))
            assert len(res_line) == len(res_label)
            res_lines.append((res_line, res_label))

        return res_lines

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


def convert_examples_to_multi_features(examples, label_list, label_list2, max_seq_length, tokenizer, task_sign="ner"):
    # load a data file into a list of "InputBatch"
    label_map = {label: i for i, label in enumerate(label_list)}
    label_map2 = {label: i for i, label in enumerate(label_list2)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(process_sent(example.text_a))
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
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

        # input_mask = [1] * len(input_ids)
        # input_mask = [0] + [1] * (len(input_ids) - 2) + [0]
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
            example.label2 = example.label2[: (max_seq_length - 2)]

        if task_sign == "cws&pos":
            label_id = [label_map["B"]] + [label_map[tmp] for tmp in example.label]
            label_id += (len(input_ids) - len(label_id)) * [label_map["B"]]
            label_id2 = [label_map2["o"]] + [label_map2[tmp] for tmp in example.label2]
            label_id2 += (len(input_ids) - len(label_id2)) * [label_map2["o"]]
        else:
            raise ValueError("task_sign not found: %s." % task_sign)

        features.append(InputFeature2(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=label_id,
                                      label_id2=label_id2,
                                      label_len=len(example.label)))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
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
        else:
            tokens_b.pop()