#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import random
import pickle
import torch
from chinese_word_segmentation.sampler import GibbsSampler
from chinese_word_segmentation.FreqDict import FreqDict
from chinese_word_segmentation.DataLoader import CWSDataLoader
from dataset_readers.bert_cws import *
from utils.tokenization import CompTokenizer, BertTokenizer


def read_from_pkl(filename):
    pkl = open(filename, 'rb')
    data = pickle.load(pkl)
    pkl.close()
    return data


def save_as_pkl(filename, data):
    output_pkl = open(filename, 'wb')
    pickle.dump(data, output_pkl)
    output_pkl.close()


def check_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def random_cutting(sents, thres=75):
    result = []
    for sent in sents:
        new_sent = "".join(sent.split(' '))
        s = str()
        for w in new_sent:
            s += w
            if random.randint(0, 100) < thres:
                s += ' '
        result.append(s)
    return result


def sampling(input, fq):
    gs = GibbsSampler()
    sents = gs.sample(freq_dict=fq, text=input, max_iters=50)
    return sents


def produce_sampling_sents(raw_sents, sample_times, freqdict, thres=8, last_time_sents=None):
    sample_sents = []
    for cnt in range(sample_times):
        sents_cut = random_cutting(raw_sents, thres=thres)
        sents = sampling(sents_cut, freqdict)
        sample_sents.append(sents)
    if last_time_sents is not None:
        print("exist last_time_sents, no random cut.")
        sents = sampling(last_time_sents, freqdict)
        sample_sents.append(sents)
    return sample_sents


def convert_feature_to_sents(dataset, tokenizer, label_list):
    label_map = {i: label for i, label in enumerate(label_list)}
    sents = []
    for input_ids, input_mask, segment_ids, label_ids, label_len in dataset:
        tokens = tokenizer.convert_ids_to_tokens(input_ids.numpy().tolist())
        raw_tokens = []
        for token in tokens:
            # skip comp-part tokens
            if token.startswith('#'):
                continue
            else:
                raw_tokens.append(token)
        raw_tokens = raw_tokens[1:label_len + 1]

        raw_label = label_ids[1:label_len + 1].numpy().tolist()
        if label_len > 1 and raw_label[1] == 1:
            raw_label[1] = 0
        raw_label = [label_map[i] for i in raw_label]

        sent = str()
        for t, l in zip(raw_tokens, raw_label):
            if l == 'B':
                sent += ' '
                sent += t
            else:
                sent += t
        sent = sent.strip()
        sents.append(sent)

    return sents


def check_word_len(sent):
    for w in sent.split():
        if len(w) > 1:
            return True
    return False


def select_train_sentences(sents_lst, freqdict, is_init=False):
    sents_one = sents_lst[0]
    sents_two = sents_lst[1]
    sents_thr = sents_lst[2]
    train_file = []
    all_one_sents = []
    remain_file = []

    for s1, s2, s3 in zip(sents_one, sents_two, sents_thr):
        if s1 == "":
            continue
        if s1 == s2 == s3:
            if check_word_len(s1):
                train_file.append(s1)
            else:
                all_one_sents.append(s1)
        else:
            remain_file.append(s1)
    if is_init:
        all_one_allow_cnt = min(len(train_file)//10, len(all_one_sents))
    else:
        all_one_allow_cnt = min(len(train_file)//2, len(all_one_sents))

    for i in range(all_one_allow_cnt):
        train_file.insert(random.randint(0, len(train_file)-1), all_one_sents.pop(0))

    remain_file.extend(all_one_sents)
    data_loader = CWSDataLoader()
    data_loader.update_freq_dict(lines=sents_one, freq_dict=freqdict)
    return train_file, remain_file


def labeled_txt_to_dataloader(config, label_list, tokenizer, sents, iter=5):
    DataProcessor = WhitespaceCWSPrecessor()
    examples = DataProcessor._create_examples(DataProcessor._read_lines(sents), "train")
    features = convert_examples_to_features(examples,
                                            label_list,
                                            config.max_seq_length,
                                            tokenizer,
                                            task_sign=config.task_name)
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    label_len = torch.tensor([f.label_len for f in features], dtype=torch.long)
    data = TensorDataset(input_ids, input_mask, segment_ids, label_ids, label_len)
    # sampler = RandomSampler(data)
    sampler = SequentialSampler(data)
    config.num_train_epochs = iter
    dataloader = DataLoader(data, sampler=sampler, batch_size=config.train_batch_size)
    num_train_steps = int(len(examples) / config.train_batch_size * config.num_train_epochs)
    return dataloader, num_train_steps
