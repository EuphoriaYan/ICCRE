#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

server_root_path = '/yjs/euphoria/GCCRE/'

import torch
import torch.nn as nn
from torch.optim import Adam
import time
from sklearn.metrics import precision_score, recall_score, f1_score

from utils.tokenization import BertTokenizer, CompTokenizer
from utils.optimization import BertAdam, warmup_linear
from utils.config import Config
from dataset_readers.bert_ner import *
from dataset_readers.bert_pos import *
from dataset_readers.bert_cws import *
from dataset_readers.bert_css import *
from models.bert.bert_multi_tagger import BertMultiTagger

logging.basicConfig()
logger = logging.getLogger(__name__)


def args_parser():
    # start parser
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument("--config_path", default="configs/bert.json", type=str)
    parser.add_argument("--data_dir", default=None, type=str, help="the input data dir")
    parser.add_argument("--data_sign", type=str, default="msra_ner")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="bert-large-uncased, bert-base-cased, bert-large-cased")
    parser.add_argument("--task_name", default=None, type=str)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--output_model_name", type=str, default="pytorch_model.bin")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval")
    parser.add_argument("--use_server", action="store_true")

    # # other parameters
    parser.add_argument("--device", type=str, default="cpu")
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

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def merge_config(args_config):
    if args_config.use_server:
        args_config.config_path = server_root_path + args_config.config_path
        args_config.bert_model = server_root_path + args_config.bert_model
        args_config.data_dir = server_root_path + args_config.data_dir
        args_config.output_dir = server_root_path + args_config.output_dir
    model_config_path = args_config.config_path
    model_config = Config.from_json_file(model_config_path)
    model_config.update_args(args_config)
    model_config.print_config()
    return model_config


def load_data(config):
    # load some data and processor

    data_processor = ZuozhuanPosProcessor_multi_output()
    cws_labels = data_processor.get_cws_labels()
    pos_labels = data_processor.get_pos_labels()

    if 'wcm' in config.bert_model:
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
        features = convert_examples_to_multi_features(examples, cws_labels, pos_labels,
                                                      config.max_seq_length,
                                                      tokenizer,
                                                      task_sign=config.task_name)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        label_ids2 = torch.tensor([f.label_id2 for f in features], dtype=torch.long)
        label_len = torch.tensor([f.label_len for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids, label_ids2, label_len)
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

    num_train_steps = int(len(train_examples) / config.train_batch_size * config.num_train_epochs)
    print("Train_examples: ", len(train_examples))
    print("Dev_examples: ", len(dev_examples))
    print("Test_examples: ", len(test_examples))
    print("Total train steps: ", num_train_steps)
    sys.stdout.flush()
    return train_dataloader, dev_dataloader, test_dataloader, num_train_steps, cws_labels, pos_labels


def load_model(config, num_train_steps, cws_labels, pos_labels):
    # device = torch.device(torch.cuda.is_available())

    if config.use_server and torch.cuda.is_available():
        if config.use_multi_gpu:
            n_gpu = torch.cuda.device_count()
            device = torch.device("cuda")
        else:
            n_gpu = 1
            device = torch.device(config.device)
    else:
        device = torch.device("cpu")
        n_gpu = 0

    model = BertMultiTagger(config, cws_labels=len(cws_labels), pos_labels=len(pos_labels))

    if config.use_server:
        model.to(device)
        print('using device:' + config.device)

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


def train(model, optimizer, train_dataloader, dev_dataloader, test_dataloader,
          config, device, n_gpu, cws_labels, pos_labels):
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    best_loss = 1e10
    cws_best_precision = 0
    cws_best_recall = 0
    cws_best_f1 = 0

    pos_best_precision = 0
    pos_best_recall = 0
    pos_best_f1 = 0

    model.train()
    train_start = time.time()
    for idx in range(int(config.num_train_epochs)):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        print("#######" * 7)
        print("EPOCH: ", str(idx))

        epoch_start = time.time()
        print("EPOCH: %s/%s" % (str(idx), config.num_train_epochs))

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, label_ids2, label_len = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids, label_ids2)
            if n_gpu > 1:
                loss = loss.mean()

            model.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)

            tr_loss += loss.item()
            nb_tr_steps += 1

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                # optimizer.zero_grad()
                global_step += 1

            if (global_step + 1) % config.checkpoint == 0:
                print("-*-" * 15)
                print("current training loss is : ")
                print(loss.item())

                if config.do_eval:
                    dev_result = eval_checkpoint(
                        model_object=model,
                        eval_dataloader=dev_dataloader,
                        device=device,
                        cws_labels=cws_labels,
                        pos_labels=pos_labels,
                    )
                    better_flag = False
                    loss, c_p, c_r, c_f1, p_p, p_r, p_f1 = dev_result
                    print("......" * 10)
                    print("Loss: Total")
                    print(loss)
                    print("......" * 10)
                    print("CWS: precision, recall, f1")
                    print(c_p, c_r, c_f1)
                    print("......" * 10)
                    print("POS: precision, recall, f1")
                    print(p_p, p_r, p_f1)
                    if loss < best_loss:
                        best_loss = loss
                    if c_f1 + p_f1 > cws_best_f1 + pos_best_f1:
                        cws_best_precision = c_p
                        cws_best_recall = c_r
                        cws_best_f1 = c_f1
                        pos_best_precision = p_p
                        pos_best_recall = p_r
                        pos_best_f1 = p_f1
                        better_flag = True

                    if config.export_model and better_flag:
                        # export a better model
                        model_to_save = model.module if hasattr(model, "module") else model
                        output_model_file = os.path.join(config.output_dir, config.output_model_name)
                        torch.save(model_to_save.state_dict(), output_model_file)
                    # end of if do_eval

                print("-*-" * 15, flush=True)
                model.train()
                # end of checkpoint

        epoch_finish = time.time()
        print("EPOCH: %d; TIME: %.2fs" % (idx, epoch_finish - epoch_start), flush=True)
        # end of epoch

    train_finish = time.time()
    print("TOTAL_TIME: %.2fs" % (train_finish - train_start))

    print("=&=" * 15)
    print("CWS+POS: best loss ")
    print(best_loss)
    print("CWS: best p, r, f1 ")
    print(cws_best_precision, cws_best_recall, cws_best_f1)
    print("POS: best p, r, f1 ")
    print(pos_best_precision, pos_best_recall, pos_best_f1)
    print("=&=" * 15)


def eval_checkpoint(model_object, eval_dataloader, device, cws_labels, pos_labels):
    # input_dataloader type can only be one of dev_dataloader, test_dataloader
    model_object.eval()
    cws_label2idx = {label: i for i, label in enumerate(cws_labels)}

    loss = 0
    cws_pred = []
    cws_gold = []
    pos_pred = []
    pos_gold = []
    lst_len = []
    eval_steps = 0

    for input_ids, input_mask, segment_ids, label_ids, label_ids2, label_len in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        label_ids2 = label_ids2.to(device)

        with torch.no_grad():
            tmp_eval_loss = model_object(input_ids, segment_ids, input_mask, label_ids, label_ids2)
            cws_logits, pos_logits = model_object(input_ids, segment_ids, input_mask)

        label_ids = label_ids.tolist()
        label_ids2 = label_ids2.tolist()
        label_len = label_len.tolist()

        loss += tmp_eval_loss.mean().item()

        cws_pred.extend(cws_logits)
        pos_pred.extend(pos_logits)
        cws_gold.extend(label_ids)
        pos_gold.extend(label_ids2)
        lst_len.extend(label_len)
        eval_steps += 1

    average_loss = round(loss / eval_steps, 4)
    pred = []
    gold = []
    for c_pred, c_gold, l in zip(cws_pred, cws_gold, lst_len):
        c_pred = c_pred[1:l + 1]
        c_gold = c_gold[1:l + 1]
        pred.extend(c_pred)
        gold.extend(c_gold)
    c_p = precision_score(gold, pred, average='binary', pos_label=cws_label2idx['B'])
    c_r = recall_score(gold, pred, average='binary', pos_label=cws_label2idx['B'])
    c_f1 = 2 * (c_p * c_r) / (c_p + c_r)

    pred = []
    gold = []
    for p_pred, p_gold, l in zip(pos_pred, pos_gold, lst_len):
        p_pred = p_pred[1:l + 1]
        p_gold = p_gold[1:l + 1]
        pred.extend(p_pred)
        gold.extend(p_gold)
    p_p = precision_score(gold, pred, average='macro')
    p_r = recall_score(gold, pred, average='macro')
    p_f1 = 2 * (p_p * p_r) / (p_p + p_r)

    c_f1, c_p, c_r = round(c_f1, 4), round(c_p, 4), round(c_r, 4)
    p_f1, p_p, p_r = round(p_f1, 4), round(p_p, 4), round(p_r, 4)
    return average_loss, c_p, c_r, c_f1, p_p, p_r, p_f1


def main():
    args_config = args_parser()
    config = merge_config(args_config)
    train_loader, dev_loader, test_loader, num_train_steps, cws_labels, pos_labels = load_data(config)
    model, optimizer, device, n_gpu = load_model(config, num_train_steps, cws_labels, pos_labels)
    train(model, optimizer, train_loader, dev_loader, test_loader, config, device, n_gpu, cws_labels, pos_labels)


if __name__ == "__main__":
    main()
