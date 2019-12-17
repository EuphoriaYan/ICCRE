#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

use_server = True
server_root_path = '/yjs/euphoria/GCCRE/'
server_cuda_device = 'cuda:2'

import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from models.gccre_bert.gccre_bert_tagger import GccreBertTagger

import csv
import logging
import argparse
import numpy as np
from tqdm import tqdm

from utils.tokenization import BertTokenizer
from utils.optimization import BertAdam, warmup_linear
from utils.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from utils.config import Config
from dataset_readers.bert_ner import *
from dataset_readers.bert_pos import *
from dataset_readers.bert_cws import *
from models.bert.bert_tagger import BertTagger
from dataset_readers.bert_data_utils import convert_examples_to_features
from utils.metrics.tagging_evaluate_funcs import compute_performance

logging.basicConfig()
logger = logging.getLogger(__name__)


def args_parser():
    # start parser
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument("--config_path", default="configs/gccre_bert.json", type=str)
    parser.add_argument("--data_dir", default=None, type=str, help="the input data dir")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="bert-large-uncased, bert-base-cased, bert-large-cased")
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
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--dev_batch_size", default=32, type=int)
    parser.add_argument("--checkpoint", default=100, type=int)
    parser.add_argument("--test_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
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

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    return args


def load_data(config):
    # load some data and processor
    data_processor_list = {
        "msra_ner": MsraNERProcessor,
        "resume_ner": ResumeNERProcessor,
        "ontonotes_ner": OntoNotesNERProcessor,
        "ctb5_pos": Ctb5POSProcessor,
        "ctb6_pos": Ctb6CWSProcessor,
        "ctb6_cws": Ctb6CWSProcessor,
        "ud1_pos": Ud1POSProcessor,
        "pku_cws": PkuCWSProcessor,
        "msr_cws": MsrCWSProcessor,
        "personal_cws": PersonalCWSProcessor
    }

    if config.data_sign not in data_processor_list:
        raise ValueError("Data_sign not found: %s".format(config.data_sign))

    data_processor = data_processor_list[config.data_sign]()

    label_list = data_processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)

    # load data examples
    train_examples = data_processor.get_train_examples(config.data_dir)
    dev_examples = data_processor.get_dev_examples(config.data_dir)
    test_examples = data_processor.get_test_examples(config.data_dir)

    def generate_data(examples):
        features = convert_examples_to_features(examples, label_list, config.max_seq_length, tokenizer,
                                                task_sign=config.task_name)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
        # sampler = DistributedSampler(data)
        sampler = RandomSampler(data)
        return data, sampler

    # convert data example into features
    train_data, train_sampler = generate_data(train_examples)
    dev_data, dev_sampler = generate_data(dev_examples)
    test_data, test_sampler = generate_data(test_examples)

    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=config.train_batch_size)

    dev_dataloader = DataLoader(dev_data,
                                sampler=dev_sampler,
                                batch_size=config.dev_batch_size)

    test_dataloader = DataLoader(test_data,
                                 sampler=test_sampler,
                                 batch_size=config.test_batch_size)

    num_train_steps = int(len(train_examples) / config.train_batch_size * config.num_train_epochs)
    return train_dataloader, dev_dataloader, test_dataloader, num_train_steps, label_list


def load_model(config, num_train_steps, label_list):

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

    model = GccreBertTagger(config, num_labels=len(label_list))
    # model = BertTagger(config, num_labels=13)
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
                         t_total=num_train_steps)

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

    for idx in range(int(config.num_train_epochs)):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        print("#######" * 7)
        print("EPOCH: ", str(idx))
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss, gccre_loss = model(input_ids, segment_ids, input_mask, label_ids)
            if n_gpu > 1:
                loss = loss.mean()
                gccre_loss = loss.mean()

            if global_step < config.gccre_warmup:
                sum_loss = loss + config.gccre_ratio * gccre_loss
            else:
                sum_loss = loss + config.gccre_ratio * gccre_loss * config.gccre_decay ** (idx + 1 + global_step // 5)

            if idx >= 1:
                sum_loss = loss

            sum_loss.backward()

            tr_loss += loss.item()

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if nb_tr_steps % config.checkpoint == 0 and config.do_eval:
                print("-*-" * 15)
                print("current training loss is : ")
                print("taggging loss, gccre loss")
                print(loss.item(), gccre_loss.item())
                tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1 = eval_checkpoint(model,
                                                                                                   dev_dataloader,
                                                                                                   config, device,
                                                                                                   n_gpu, label_list,
                                                                                                   eval_sign="dev")
                print("......" * 10)
                print("DEV: loss, acc, precision, recall, f1")
                print(tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1)

                if tmp_dev_f1 > dev_best_f1 or tmp_dev_acc > dev_best_acc:
                    dev_best_acc = tmp_dev_acc
                    dev_best_loss = tmp_dev_loss
                    dev_best_precision = tmp_dev_prec
                    dev_best_recall = tmp_dev_rec
                    dev_best_f1 = tmp_dev_f1

                    tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1 = eval_checkpoint(model,
                                                                                                            test_dataloader,
                                                                                                            config,
                                                                                                            device,
                                                                                                            n_gpu,
                                                                                                            label_list,
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
                            output_model_file = os.path.join(config.output_dir, "bert_finetune_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)

                print("-*-" * 15)
                model.train()

    # export a trained mdoel
    model_to_save = model
    output_model_file = os.path.join(config.output_dir, "bert_model.bin")
    if config.export_model:
        torch.save(model_to_save.state_dict(), output_model_file)

    print("=&=" * 15)
    print("DEV: current best precision, recall, f1, acc, loss ")
    print(dev_best_precision, dev_best_recall, dev_best_f1, dev_best_acc, dev_best_loss)
    print("TEST: current best precision, recall, f1, acc, loss ")
    print(test_best_precision, test_best_recall, test_best_f1, test_best_acc, test_best_loss)
    print("=&=" * 15)


def eval_checkpoint(model_object, eval_dataloader, config, device, n_gpu, label_list, eval_sign="dev"):
    # input_dataloader type can only be one of dev_dataloader, test_dataloader
    model_object.eval()

    idx2label = {i: label for i, label in enumerate(label_list)}

    eval_loss = 0
    pred_lst = []
    mask_lst = []
    gold_lst = []
    eval_steps = 0

    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss, gccre_loss = model_object(input_ids, segment_ids, input_mask, label_ids)
            logits, gccre_loss = model_object(input_ids, segment_ids, input_mask)

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

    eval_accuracy, eval_precision, eval_recall, eval_f1 = compute_performance(
        pred_lst,
        gold_lst,
        mask_lst,
        label_list,
        dims=2)

    average_loss = round(eval_loss / eval_steps, 4)
    eval_f1 = round(eval_f1, 4)
    eval_precision = round(eval_precision, 4)
    eval_recall = round(eval_recall, 4)
    eval_accuracy = round(eval_accuracy, 4)

    return average_loss, eval_accuracy, eval_precision, eval_recall, eval_f1


def merge_config(args_config):
    if use_server:
        args_config.config_path = server_root_path + args_config.config_path
        args_config.bert_model = server_root_path + args_config.bert_model
        args_config.data_dir = server_root_path + args_config.data_dir
        args_config.output_dir = server_root_path + args_config.output_dir
    model_config_path = args_config.config_path
    model_config = Config.from_json_file(model_config_path)
    if use_server:
        model_config.gccre_config.bert_model = server_root_path + model_config.gccre_config.bert_model
        model_config.gccre_config.pretrained_char_embedding_path = server_root_path + model_config.gccre_config.pretrained_char_embedding_path
        model_config.gccre_config.gccre_pos_emb_model = server_root_path + model_config.gccre_config.gccre_pos_emb_model
    model_config.update_args(args_config)
    model_config.print_config()
    return model_config


def main():
    args_config = args_parser()
    config = merge_config(args_config)
    train_loader, dev_loader, test_loader, num_train_steps, label_list = load_data(config)
    model, optimizer, device, n_gpu = load_model(config, num_train_steps, label_list)
    train(model, optimizer, train_loader, dev_loader, test_loader, config, device, n_gpu, label_list)


if __name__ == "__main__":
    main()
