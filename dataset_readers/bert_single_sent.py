# encoding: utf-8


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
import json
from tqdm import tqdm


from dataset_readers.bert_data_utils import *



def read_json(file):
    data = []
    print("read json:")
    with open(file, 'r', encoding='utf8') as f:
        for line in tqdm(f.readlines()):
            data.append(json.loads(line.strip()))
    return data



class ChinaNewsProcessor(DataProcessor):
    """Processor for the dbqa data set """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            read_json(os.path.join(data_dir, "valid.json")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            read_json(os.path.join(data_dir, "test.json")),
            "test_matched")

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4", "5", "6", "7"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            # text may have multiple fields, join and separate by [SEP]
            text_a = line["sentence"]
            label = line["gold_label"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class DianPingProcessor(DataProcessor):
    """Processor for the dbqa data set """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            read_json(os.path.join(data_dir, "valid.json")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            read_json(os.path.join(data_dir, "test.json")),
            "test_matched")

    def get_labels(self):
        """See base class."""
        return ["1", "2"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            # text may have multiple fields, join and separate by [SEP]
            text_a = line["sentence"]
            label = line["gold_label"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class JDFullProcessor(DataProcessor):
    """Processor for the dbqa data set """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            read_json(os.path.join(data_dir, "valid.json")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            read_json(os.path.join(data_dir, "test.json")),
            "test_matched")

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4", "5"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            # text may have multiple fields, join and separate by [SEP]
            text_a = line["sentence"]
            label = line["gold_label"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples



class JDBinaryProcessor(DataProcessor):
    """Processor for the dbqa data set """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "valid.csv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv")),
            "test_matched")

    def get_labels(self):
        """See base class."""
        return ["1", "2"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[1:]):
            guid = "%s-%s" % (set_type, i)
            # text may have multiple fields, join and separate by [SEP]
            text_a = "[SEP]".join(line[1:]).strip("[SEP]")
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class FuDanProcessor(DataProcessor):
    """Processor for the dbqa data set """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            read_json(os.path.join(data_dir, "valid.json")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            read_json(os.path.join(data_dir, "test.json")),
            "test_matched")

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4", "5"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            # text may have multiple fields, join and separate by [SEP]
            text_a = line["doc"][:512]
            label = str(line["gold_label"])
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples



class ChnSentiCorpProcessor(DataProcessor):
    """Processor for the ChnSentiCorp data set """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            label = line[0]
            text_a = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples



class ifengProcessor(DataProcessor):
    """Processor for the ifeng data set """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "valid.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4", "5"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            label = line[0]
            text_a = "[SEP]".join(line[1:]).strip("[SEP]")
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class DzgProcessor(DataProcessor):
    """Processor for the dzg data set """

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.txt")), "test")

    def get_labels(self):
        return ["art", "bud", "med", "tao"]

    def _read_tsv(self, input_file, quotechar=None):
        # reads a tab separated value file.
        with open(input_file, "r", encoding='utf8', errors='ignore') as f:
            reader = csv.reader(f, delimiter="|", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "{}_{}".format("dzg.clf", str(i))
            text_a = line[0].strip()
            label = line[1]
            if not text_a or label is None:
                print(guid)
                print(text_a)
                print(label)
                continue
            if label not in self.get_labels():
                raise ValueError('label {} is wrong'.format(label))
            examples.append(InputExample(guid=guid, text_a=text_a, label=[label]))
        return examples
