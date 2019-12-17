#!/usr/bin/env python3 
# -*- coding: utf-8 -*-


import os 
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

use_server = True
server_root_path = '/yjs/euphoria/GCCRE/'
server_cuda_device = 'cuda:3'

import torch
import torch.optim as optim
import torch.autograd as autograd


import gc
import time
import logging
import argparse
import datetime
import numpy as np
import random

from models.latticeLSTM.utils.data import Data
from models.latticeLSTM.utils.metric import get_ner_fmeasure
from models.latticeLSTM.model.bilstmcrf import BiLSTMCRF
from dataset_readers.lstm_data_utils import *
from dataset_readers.lstm_cws import *
from utils.config import Config
from utils.tokenization import BertTokenizer
from utils.optimization import BertAdam
from utils.metrics.tagging_evaluate_funcs import compute_performance

logging.basicConfig()
logger = logging.getLogger(__name__)


def args_parser():
    parser = argparse.ArgumentParser(description='Lattice-LSTM-CRF')

    # required parameters
    parser.add_argument("--config_path", default="configs/lattice_lstm.json", type=str)
    parser.add_argument("--data_dir", default=None, type=str, help="the input data dir")
    parser.add_argument("--task_name", default=None, type=str)

    # other parameters
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--use_multi_gpu", type=bool, default=False)
    parser.add_argument("--max_seq_length", default=128,
                        type=int, help="the maximum total input sequence length after ")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--checkpoint", default=100, type=int)
    parser.add_argument("--learning_rate", default=1e-2, type=float)
    parser.add_argument("--num_train_epochs", default=4.0, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    # parser.add_argument("--bert_frozen", type=bool, default=False)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--export_model", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--data_sign", type=str, default="msra_ner")
    parser.add_argument("--classifier_sign", type=str, default="multi_nonlinear")
    args = parser.parse_args()
    args.batch_size = args.batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    return args


def merge_config(args_config):
    if use_server:
        args_config.config_path = server_root_path + args_config.config_path
        args_config.data_dir = server_root_path + args_config.data_dir
        args_config.output_dir = server_root_path + args_config.output_dir
    model_config_path = args_config.config_path
    model_config = Config.from_json_file(model_config_path)
    if use_server:
        model_config.gccre_config.pretrained_char_embedding_path = server_root_path + model_config.gccre_config.pretrained_char_embedding_path
    model_config.update_args(args_config)
    model_config.print_config()
    return model_config


def load_data(config):

    data_processor_list = {
        "personal_cws": PersonalCWSLSTMProcessor
    }

    if config.data_sign not in data_processor_list:
        raise ValueError("Data_sign not found: %s".format(config.data_sign))

    data_processor = data_processor_list[config.data_sign]()
    label_list = data_processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(config.bert_tokenizer, do_lower_case=True)

    # load data examples
    train_examples = data_processor.get_train_examples(config.data_dir)
    dev_examples = data_processor.get_dev_examples(config.data_dir)
    test_examples = data_processor.get_test_examples(config.data_dir)

    # load components dictionary
    component_dict_path = os.path.join(root_path, 'chaizi')
    config.component_dict = collections.OrderedDict()
    with open(os.path.join(component_dict_path, 'chaizi-jt.txt'), encoding='utf-8') as cz_jt:
        for line in cz_jt:
            item_list = line.strip().split('\t')
            config.component_dict[item_list[0]] = list(item_list[1].split(' '))

    def generate_data(examples):
        features = convert_examples_to_gccre_features_lstm(examples,
                                                           component_dict=config.component_dict,
                                                           label_list=label_list,
                                                           max_seq_length=config.max_seq_length,
                                                           tokenizer=tokenizer,
                                                           task_sign=config.task_name)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        component_ids = torch.tensor([f.component_ids for f in features], dtype=torch.long)
        component_len = torch.tensor([f.component_len for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, component_ids, component_len, label_ids)
        # sampler = DistributedSampler(data)
        sampler = RandomSampler(data)
        return data, sampler

    train_data, train_sampler = generate_data(train_examples)
    dev_data, dev_sampler = generate_data(dev_examples)
    test_data, test_sampler = generate_data(test_examples)

    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=config.batch_size)

    dev_dataloader = DataLoader(dev_data,
                                sampler=dev_sampler,
                                batch_size=config.batch_size)

    test_dataloader = DataLoader(test_data,
                                 sampler=test_sampler,
                                 batch_size=config.batch_size)

    num_train_steps = int(len(train_examples) / config.batch_size * config.num_train_epochs)
    return train_dataloader, dev_dataloader, test_dataloader, num_train_steps, label_list


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        # logger.info "p:",pred, pred_tag.tolist()
        # logger.info "g:", gold, gold_tag.tolist()
        assert(len(pred)==len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def load_data_setting(save_dir):
    with open(save_dir, 'rb') as fp:
        data = torch.load(fp)
    logger.info("Data setting loaded from file: " + save_dir)
    data.show_data_summary()
    return data


def evaluate(data, model, name):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        logger.info("Error: wrong evaluate name," + name)
    pred_results = []
    gold_results = []
    model.eval()
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//data.HP_batch_size+1

    for batch_id in range(total_batch):
        start = batch_id*data.HP_batch_size
        end = (batch_id+1)*data.HP_batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        gaz_list, batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, True)
        tag_seq = model(gaz_list,batch_word, batch_biword, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
        # logger.info("tag_seq", tag_seq)
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    return speed, acc, p, r, f, pred_results  


def batchify_with_label(input_batch_list, gpu, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words,biwords,chars,gaz, labels],[words,biwords,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len) 
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order 
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len) 
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    biwords = [sent[1] for sent in input_batch_list]
    chars = [sent[2] for sent in input_batch_list]
    gazs = [sent[3] for sent in input_batch_list]
    labels = [sent[4] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    biword_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).byte()
    for idx, (seq, biseq, label, seqlen) in enumerate(zip(words, biwords, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        biword_seq_tensor[idx, :seqlen] = torch.LongTensor(biseq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    biword_seq_tensor = biword_seq_tensor[word_perm_idx]
    ## not reorder label
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(list(map(max, length_list)))
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)), volatile =  volatile_flag).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # logger.info len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    
    #  keep the gaz_list in orignial order
    
    gaz_list = [gazs[i] for i in word_perm_idx]
    gaz_list.append(volatile_flag)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        biword_seq_tensor = biword_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return gaz_list, word_seq_tensor, biword_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


def load_model(config, num_train_steps, label_list):
    # device = torch.device(torch.cuda.is_available())
    if use_server and torch.cuda.is_available():
        if config.use_multi_gpu:
            n_gpu = torch.cuda.device_count()
            device = torch.device("cuda")
        else:
            n_gpu = 1
            device = torch.device(server_cuda_device)
    else:
        device = torch.device("cpu")
        n_gpu = 0

    model = BiLSTMCRF(config, num_labels=len(label_list))

    if use_server:
        model.to(device)
        print('using device:' + server_cuda_device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=config.warmup_proportion,
                         t_total=num_train_steps,
                         max_grad_norm=config.clip_grad)

    return model, optimizer, device, n_gpu


def train(model, optimizer, train_dataloader, dev_dataloader, test_dataloader, config, device, n_gpu, label_list):

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    dev_best_acc = 0
    dev_best_precision = 0
    dev_best_recall = 0
    dev_best_f1 = 0
    dev_best_loss = 10000000000000

    test_best_acc = 0
    test_best_precision = 0
    test_best_recall = 0
    test_best_f1 = 0
    test_best_loss = 1000000000000000

    model.train()
    train_start = time.time()

    #  start training
    for idx in range(int(config.num_train_epochs)):
        epoch_start = time.time()
        print(("Epoch: %s/%s" .format(idx, config.num_train_epochs)))

        tr_loss = 0
        nb_tr_examples = 0
        nb_tr_steps = 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, component_ids, component_len, label_ids = batch
            loss = model(input_ids, input_mask, component_ids, component_len, label_ids)
            if n_gpu > 1:
                loss = loss.mean()

            model.zero_grad()
            loss.backward()

            tr_loss += loss.item()

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                # optimizer.zero_grad()
                global_step += 1

            if nb_tr_steps % config.checkpoint == 0:
                print("-*-" * 15)
                print("current training loss is : ")
                print(loss.item())
                tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1 = eval_checkpoint(model,
                                                                                                   dev_dataloader,
                                                                                                   config, device,
                                                                                                   n_gpu, label_list,
                                                                                                   eval_sign="dev")

                if tmp_dev_f1 > dev_best_f1 or tmp_dev_acc > dev_best_acc:
                    dev_best_acc = tmp_dev_acc
                    dev_best_loss = tmp_dev_loss
                    dev_best_precision = tmp_dev_prec
                    dev_best_recall = tmp_dev_rec
                    dev_best_f1 = tmp_dev_f1

                    tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1 = eval_checkpoint(model,
                                                                                                            test_dataloader,
                                                                                                            config, device,
                                                                                                            n_gpu, label_list,
                                                                                                            eval_sign="test")
                    print("......" * 10)
                    print("TEST: loss, acc, precision, recall, f1")
                    print(tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1)

                    if tmp_test_f1 > test_best_f1 or tmp_test_acc > test_best_acc:
                        test_best_acc = tmp_test_acc
                        test_best_loss = tmp_test_loss
                        test_best_precision = tmp_test_prec
                        test_best_recall = tmp_test_rec
                        test_best_f1 = tmp_test_f1

                        # export model
                        if config.export_model:
                            model_to_save = model.module if hasattr(model, "module") else model
                            output_model_file = os.path.join(config.output_dir, "lattice_lstm_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)

                print("-*-" * 15)
                model.train()
            # end of checkpoint process.

        epoch_finish = time.time()
        print("EPOCH: %d; TIME: %.2fs".format(idx, epoch_finish - epoch_start))
    # end of training.

    train_finish = time.time()
    print("TOTAL_TIME: %.2fs".format(train_finish - train_start))

    # export a trained mdoel
    model_to_save = model
    output_model_file = os.path.join(config.output_dir, "bert_model.bin")
    if config.export_model == "True":
        torch.save(model_to_save.state_dict(), output_model_file)

    print("=&=" * 15)
    print("DEV: current best precision, recall, f1, acc, loss ")
    print(dev_best_precision, dev_best_recall, dev_best_f1, dev_best_acc, dev_best_loss)
    print("TEST: current best precision, recall, f1, acc, loss ")
    print(test_best_precision, test_best_recall, test_best_f1, test_best_acc, test_best_loss)
    print("=&=" * 15)


def eval_checkpoint(model, eval_dataloader, config, device, n_gpu, label_list, eval_sign="dev"):
    # input_dataloader type can only be one of dev_dataloader, test_dataloader
    model.eval()

    idx2label = {i: label for i, label in enumerate(label_list)}

    eval_loss = 0
    pred_lst = []
    mask_lst = []
    gold_lst = []
    eval_steps = 0

    for input_ids, input_mask, component_ids, component_len, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        component_ids = component_ids.to(device)
        component_len = component_len.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, input_mask, component_ids, component_len, label_ids)
            logits = model(input_ids, input_mask, component_ids, component_len)

        logits = logits.detach().cpu().numpy()
        # logits = np.argmax(logits, axis=-1)
        label_ids = label_ids.to("cpu").numpy()
        input_mask = input_mask.to("cpu").numpy()
        reshape_lst = label_ids.shape
        logits = np.reshape(logits, (reshape_lst[0], reshape_lst[1], -1))
        logits = np.argmax(logits, axis=-1)

        logits = logits.tolist()
        label_ids = label_ids.tolist()
        input_mask = input_mask.tolist()

        eval_loss += tmp_eval_loss.mean().item()

        pred_lst += logits
        gold_lst += label_ids
        mask_lst += input_mask
        eval_steps += 1

    eval_accuracy, eval_precision, eval_recall, eval_f1 = compute_performance(pred_lst, gold_lst,
                                                                              mask_lst, label_list,
                                                                              dims=2)

    average_loss = round(eval_loss / eval_steps, 4)
    eval_f1 = round(eval_f1, 4)
    eval_precision = round(eval_precision, 4)
    eval_recall = round(eval_recall, 4)
    eval_accuracy = round(eval_accuracy, 4)

    return average_loss, eval_accuracy, eval_precision, eval_recall, eval_f1


def load_model_decode(save_dir, data):
    logger.info("Load Model from file: " + save_dir)
    model = BiLSTMCRF(data)
    model.load_state_dict(torch.load(save_dir))
    logger.info(F"Decode dev data ...")
    start_time = time.time()
    speed, acc, p, r, f, pred_results = evaluate(data, model, 'dev')
    end_time = time.time()
    time_cost = end_time - start_time
    logger.info(("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%('dev', time_cost, speed, acc, p, r, f)))

    logger.info(F"Decode test data ...")
    start_time = time.time()
    speed, acc, p, r, f, pred_results = evaluate(data, model, 'test')
    end_time = time.time()
    time_cost = end_time - start_time
    logger.info(("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%('test', time_cost, speed, acc, p, r, f)))


if __name__ == '__main__':
    config = args_parser()
    config = merge_config(config)
    train_loader, dev_loader, test_loader, num_train_steps, label_list = load_data(config)
    model, optimizer, device, n_gpu = load_model(config, num_train_steps, label_list)
    train(model, optimizer, train_loader, dev_loader, test_loader, config, device, n_gpu, label_list)
