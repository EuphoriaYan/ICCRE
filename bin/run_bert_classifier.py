#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
import torch.nn as nn
from torch.optim import Adam
import math
import time

from utils.tokenization import BertTokenizer, CompTokenizer
from utils.optimization import BertAdam
from utils.config import Config
from dataset_readers.bert_sent_pair import *
from dataset_readers.bert_single_sent import *
from models.bert.bert_classifier import BertClassifier
from dataset_readers.bert_data_utils import convert_examples_to_features
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

logging.basicConfig()
logger = logging.getLogger(__name__)


def args_parser():
    # start parser
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument("--config_path", default="configs/bert.json", type=str)
    parser.add_argument("--data_dir", default=None, type=str, help="the input data dir")
    parser.add_argument("--data_sign", type=str, default="dzg_clf")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="bert-large-uncased, bert-base-cased, bert-large-cased")
    parser.add_argument("--task_name", default=None, type=str)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--output_model_name", type=str, default="pytorch_model.bin")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval")
    parser.add_argument("--use_comp", action="store_true")

    # # other parameters
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--use_multi_gpu", type=bool, default=False)
    parser.add_argument("--max_seq_length", default=128,
                        type=int, help="the maximum total input sequence length after ")
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--dev_batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)

    parser.add_argument("--checkpoint", default=500, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=4.0, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--export_model", type=bool, default=True)
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
    # data_processor = MsraNerProcessor()

    data_processor_list = {
        "chn_sent": ChnSentiCorpProcessor,
        "ifeng_clf": ifengProcessor,
        "dzg_clf": DzgProcessor,
    }

    if config.data_sign not in data_processor_list:
        raise ValueError("Data_sign not found: %s" % config.data_sign)

    data_processor = data_processor_list[config.data_sign]()

    label_list = data_processor.get_labels()
    if config.use_comp:
        tokenizer = CompTokenizer(config.bert_model)
    else:
        tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)

    # load data exampels
    train_examples = data_processor.get_train_examples(config.data_dir)
    dev_examples = data_processor.get_dev_examples(config.data_dir)
    test_examples = data_processor.get_test_examples(config.data_dir)

    def generate_data(examples, set_type="train"):
        print(set_type + " examples")
        for i in range(min(len(examples), 3)):
            print(examples[i])
        sys.stdout.flush()
        features = convert_examples_to_features(examples, label_list, config.max_seq_length, tokenizer,
                                                task_sign=config.task_name)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        char_mask = torch.tensor([f.char_mask for f in features], dtype=torch.bool)
        label_len = torch.tensor([f.label_len for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids, char_mask, label_len)
        # sampler = DistributedSampler(data)
        sampler = RandomSampler(data)
        return data, sampler

    # convert data example into featrues
    train_data, train_sampler = generate_data(train_examples, "train")
    dev_data, dev_sampler = generate_data(dev_examples, "dev")
    test_data, test_sampler = generate_data(test_examples, "test")

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.train_batch_size)

    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=config.dev_batch_size)

    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=config.test_batch_size)

    num_train_steps = int(math.ceil(len(train_examples) / config.train_batch_size) * config.num_train_epochs)
    print("Train_examples: ", len(train_examples))
    print("Dev_examples: ", len(dev_examples))
    print("Test_examples: ", len(test_examples))
    print("Total train steps: ", num_train_steps)
    sys.stdout.flush()

    return train_dataloader, dev_dataloader, test_dataloader, num_train_steps, label_list


def load_model(config, num_train_steps, label_list):

    if torch.cuda.is_available():
        if config.use_multi_gpu:
            n_gpu = torch.cuda.device_count()
        else:
            n_gpu = 1
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")
        n_gpu = 0

    model = BertClassifier(config, num_labels=len(label_list))
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # prepare  optimzier
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


def train(model, optimizer, train_dataloader, dev_dataloader, test_dataloader,
          config, device, n_gpu, label_list):
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    dev_best_acc = 0
    dev_best_f1 = 0
    dev_best_loss = 10000000000000

    model.train()
    optimizer.zero_grad()
    train_start = time.time()

    for idx in range(int(config.num_train_epochs)):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        print("#######" * 7)
        epoch_start = time.time()
        print("EPOCH: %s/%s" % (str(idx + 1), config.num_train_epochs))

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, char_mask, label_len = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            if n_gpu > 1:
                loss = loss.mean()

            loss.backward()

            tr_loss += loss.item()

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if (nb_tr_steps+1) % config.checkpoint == 0:
                print("-*-" * 15)
                print("current training loss is : ")
                print(loss.item())

                dev_loss, dev_acc, dev_f1 = eval_checkpoint(model,
                                                            dev_dataloader,
                                                            config, device,
                                                            n_gpu, label_list,
                                                            eval_sign="dev")
                print("......" * 10)
                print("DEV: loss, acc, f1")
                print(dev_loss, dev_acc, dev_f1)

                if dev_f1 > dev_best_f1 or dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                    dev_best_loss = dev_loss
                    dev_best_f1 = dev_f1

                    # export model
                    if config.export_model:
                        model_to_save = model.module if hasattr(model, "module") else model
                        output_model_file = os.path.join(config.output_dir, config.output_model_name)
                        torch.save(model_to_save.state_dict(), output_model_file)

                print("-*-" * 15)
                model.train()
                # end of checkpoint
        epoch_finish = time.time()
        print("EPOCH: %d; TIME: %.2fs" % (idx, epoch_finish - epoch_start), flush=True)
        # end of epoch

    train_finish = time.time()
    print("TOTAL_TIME: %.2fs" % (train_finish - train_start))

    print("=&=" * 15)
    print("DEV: current best f1, acc, loss ")
    print(dev_best_f1, dev_best_acc, dev_best_loss)
    print("=&=" * 15)


def eval_checkpoint(model_object, eval_dataloader, config,
                    device, n_gpu, label_list, eval_sign="dev"):
    # input_dataloader type can only be one of dev_dataloader, test_dataloader
    model_object.eval()

    idx2label = {i: label for i, label in enumerate(label_list)}

    eval_loss = 0
    eval_steps = 0
    gold_all = []
    pred_all = []

    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, char_mask, label_len = batch

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model_object(input_ids, segment_ids, input_mask, label_ids)
            logits = model_object(input_ids, segment_ids, input_mask)

        eval_loss += tmp_eval_loss.mean().item()
        eval_steps += 1

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to("cpu").numpy()
        logits = np.argmax(logits, axis=-1)

        logits = logits.tolist()
        label_ids = label_ids.tolist()
        pred_all.extend(logits)
        gold_all.extend(label_ids)

    acc = accuracy_score(gold_all, pred_all)
    f1 = f1_score(gold_all, pred_all, average='macro')

    average_loss = round(eval_loss / eval_steps, 4)
    eval_acc = round(acc, 4)
    eval_f1 = round(f1, 4)

    return average_loss, eval_acc, eval_f1  # eval_precision, eval_recall, eval_f1


def merge_config(args_config):
    """
    if args_config.use_server:
        args_config.config_path = server_root_path + args_config.config_path
        args_config.bert_model = server_root_path + args_config.bert_model
        args_config.data_dir = server_root_path + args_config.data_dir
        args_config.output_dir = server_root_path + args_config.output_dir
    """
    model_config_path = args_config.config_path
    model_config = Config.from_json_file(model_config_path)
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
