#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

server_root_path = ''

import torch
import time
import itertools

from utils.tokenization import BertTokenizer, CompTokenizer
from utils.multistage_utils import *
from utils.optimization import BertAdam
from utils.config import Config
from dataset_readers.bert_ner import *
from dataset_readers.bert_pos import *
from dataset_readers.bert_cws import *
from models.bert.bert_tagger import BertTagger
from dataset_readers.bert_data_utils import convert_examples_to_features
from bin.eval_model import eval_checkpoint

logging.basicConfig()
logger = logging.getLogger(__name__)


def args_parser():
    # start parser
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument("--config_path", default="configs/bert.json", type=str)
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="bert-large-uncased, bert-base-cased, bert-large-cased")
    parser.add_argument("--task_name", default=None, type=str)

    # # other parameters
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--use_multi_gpu", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_seq_length", default=128,
                        type=int, help="the maximum total input sequence length after ")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval")
    parser.add_argument("--use_server", action="store_true",
                        help="Whether to use server")

    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--dev_batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)
    parser.add_argument("--checkpoint", default=500, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    # parser.add_argument("--bert_frozen", type=bool, default=False)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=77777)
    parser.add_argument("--export_model", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--raw_data", type=str, required=True)
    parser.add_argument("--book_dir", type=str, required=True)
    parser.add_argument("--use_crf", type=bool, default=True)

    parser.add_argument("--train_iterator", type=int, default=3)
    parser.add_argument("--output_model_name", type=str, default="multi_stage_pytorch.bin")
    args = parser.parse_args()

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    return args


def load_model(config, label_list):
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

    model = BertTagger(config, num_labels=len(label_list))
    # model_path = os.path.join(config.output_dir, config.ckpt_name)

    if config.use_server:
        model.to(device)
        print('using device:' + config.device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model, device, n_gpu


def merge_config(args_config):
    if args_config.use_server:
        args_config.config_path = server_root_path + args_config.config_path
        args_config.bert_model = server_root_path + args_config.bert_model
        args_config.output_dir = server_root_path + args_config.output_dir
        args_config.raw_data = server_root_path + args_config.raw_data
        args_config.book_dir = server_root_path + args_config.book_dir
    model_config_path = args_config.config_path
    model_config = Config.from_json_file(model_config_path)
    model_config.update_args(args_config)
    model_config.print_config()
    return model_config


def get_optimizer(model, config, num_train_steps):
    # get different num_train_steps optimizers
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
    return optimizer


def one_iter(bert, raw_sents, freqdict, last_time_dataset=None, sample_times=3, is_init=False, thres=75, iters=3):
    model, config, device, n_gpu, label_list, tokenizer = bert
    if last_time_dataset is None:
        sample_sents_lst = produce_sampling_sents(raw_sents, sample_times,
                                                  freqdict=freqdict,
                                                  thres=thres)
    else:
        last_time_sents = convert_feature_to_sents(last_time_dataset, tokenizer, label_list)
        # with open('data/predict.txt', 'w', encoding='utf-8') as f:
        #     for s in last_time_sents:
        #         f.write(s + '\n')
        sample_sents_lst = produce_sampling_sents(raw_sents, sample_times - 1,
                                                  freqdict=freqdict,
                                                  thres=thres,
                                                  last_time_sents=last_time_sents)

    for sample_sents in sample_sents_lst:
        print("sample sentences examples:")
        for i in range(min(3, len(sample_sents))):
            print(sample_sents[i])
        print("-*-" * 7)
    train_sents, remain_sents = select_train_sentences(sample_sents_lst, freqdict, is_init=is_init)
    print("select sentences examples:")
    for i in range(min(10, len(train_sents))):
        print(train_sents[i])
    print("=&=" * 7)
    print("Select %d sentences, remain %d sentences." % (len(train_sents), len(remain_sents)))
    train_dataloader, num_train_steps = labeled_txt_to_dataloader(config, label_list, tokenizer, train_sents, iter=iters)
    if len(train_sents) < 32:
        print("Only select %d sentences, return." % len(train_sents))
        return train_sents, remain_sents
    auto_train(model, train_dataloader, config, device, n_gpu, label_list, num_train_steps, iters=iters)
    return train_sents, remain_sents


def auto_train(model, train_dataloader, config, device, n_gpu, label_list, num_train_steps, iters=5):
    iter_cnt = 0
    config.num_train_epochs = 3
    while iter_cnt < iters:
        optimizer = get_optimizer(model, config, num_train_steps)
        loss = train(model, optimizer, train_dataloader, config, device, n_gpu, label_list)
        if loss < 5.0:
            print("Auto train complete, loss is %.4f" % loss)
            break
        elif loss < 50.0:
            continue
        else:
            iters += 1
        iter_cnt += 1
        if iter_cnt == iters:
            print("%d times of auto_train iteration. Return." % iters)
            break


def train(model, optimizer, train_dataloader, config, device, n_gpu, label_list):
    global_step = 0

    model.train()
    train_start = time.time()
    tr_loss = 0
    for idx in range(int(config.num_train_epochs)):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        print("#######" * 7)
        print("EPOCH: ", str(idx))

        epoch_start = time.time()
        print("EPOCH: %d/%d" % (idx, int(config.num_train_epochs)))

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, label_len = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            if n_gpu > 1:
                loss = loss.mean()

            model.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)

            tr_loss += loss.item()

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                # optimizer.zero_grad()
                global_step += 1

            if (global_step + 1) % config.checkpoint == 0:
                print("-*-" * 15)
                print("current batch training loss is : ")
                print(loss.item())
                print("-*-" * 15)
        print("Total training loss is: ", str(tr_loss))
        epoch_finish = time.time()
        print("EPOCH: %d; TIME: %.2fs" % (idx, epoch_finish - epoch_start), flush=True)

    train_finish = time.time()
    print("TOTAL_TIME: %.2fs" % (train_finish - train_start))

    # export a trained model
    model_to_save = model.module if hasattr(model, "module") else model
    output_model_file = os.path.join(config.output_dir, config.output_model_name)
    if config.export_model:
        torch.save(model_to_save.state_dict(), output_model_file)

    return tr_loss


def predict(model_object, device, eval_dataloader):
    model_object.eval()

    input_ids_lst = []
    input_mask_lst = []
    segment_ids_lst = []
    label_ids_lst = []
    label_len_lst = []

    for input_ids, input_mask, segment_ids, label_ids, label_len in eval_dataloader:
        input_ids_lst.append(input_ids)
        input_mask_lst.append(input_mask)
        segment_ids_lst.append(segment_ids)
        # label_ids_lst.append(label_ids)
        label_len_lst.append(label_len)

        input_ids_cuda = input_ids.to(device)
        segment_ids_cuda = segment_ids.to(device)
        input_mask_cuda = input_mask.to(device)

        with torch.no_grad():
            logits = model_object(input_ids_cuda, segment_ids_cuda, input_mask_cuda)

        logits = np.array(logits)
        logits = torch.tensor(logits, dtype=torch.long)
        label_ids_lst.append(logits)

    input_ids = torch.cat(input_ids_lst, dim=0)
    input_mask = torch.cat(input_mask_lst, dim=0)
    segment_ids = torch.cat(segment_ids_lst, dim=0)
    label_ids = torch.cat(label_ids_lst, dim=0)
    label_len = torch.cat(label_len_lst, dim=0)

    data = TensorDataset(input_ids, input_mask, segment_ids, label_ids, label_len)
    return data


def test(model, test_dataloader, config, device, n_gpu, label_list):
    test_loss, test_acc, test_prec, test_rec, test_f1 = eval_checkpoint(
        model_object=model,
        eval_dataloader=test_dataloader,
        device=device,
        label_list=label_list,
        task_sign=config.task_name
    )

    print("TEST: precision, recall, f1, acc, loss ")
    print(test_prec, test_rec, test_f1, test_acc, test_loss, '\n')
    return


def load_data(config):
    # load some data and processor

    data_processor = BookCWSProcessor()

    label_list = data_processor.get_labels()
    if 'wcm' in config.bert_model:
        tokenizer = CompTokenizer(config.bert_model)
    else:
        tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)

    # load data exampels
    test_examples_list, book_list = data_processor.get_test_examples(config.book_dir)

    cnt_B = 0
    cnt = 0
    for test_examples in test_examples_list:
        for test_line in test_examples:
            label = test_line.label
            for l in label:
                if l == 'B':
                    cnt_B += 1
                cnt += 1
    print("Total %d tags, %d tags are [B]. " % (cnt, cnt_B))

    def generate_data(examples):
        features = convert_examples_to_features(examples, label_list, config.max_seq_length, tokenizer,
                                                task_sign=config.task_name)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        label_len = torch.tensor([f.label_len for f in features], dtype=torch.long)
        data = TensorDataset(input_ids, input_mask, segment_ids, label_ids, label_len)
        # sampler = DistributedSampler(data)
        sampler = RandomSampler(data)
        return data, sampler

    # convert data example into featrues

    test_dataloader_list = []
    for test_examples in test_examples_list:
        test_data, test_sampler = generate_data(test_examples)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=config.test_batch_size)
        test_dataloader_list.append(test_dataloader)

    return test_dataloader_list, label_list, book_list


def main():
    args_config = args_parser()
    config = merge_config(args_config)

    if 'wcm' in config.bert_model:
        tokenizer = CompTokenizer(config.bert_model)
    else:
        tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)

    label_list = WhitespaceCWSPrecessor.get_labels()
    bert_model, device, n_gpu = load_model(config, label_list)
    fq = FreqDict()

    raw_sents = [s.strip() for s in open(config.raw_data, 'r', encoding='utf-8').readlines()]

    bert = bert_model, config, device, n_gpu, label_list, tokenizer

    # First Part, select sentences, train bert.
    train_sents, remain_sents = one_iter(bert, raw_sents, fq,
                                         is_init=True, thres=75)
    print("First Part Done.")
    print("-*-" * 7)
    print("=&=" * 7)
    sys.stdout.flush()

    # Second Part, use bert to predict remain sentences.
    print("\nSecond Part Start.")
    eval_dataloader, _ = labeled_txt_to_dataloader(config, label_list, tokenizer, remain_sents)
    predicted_data = predict(bert_model, device, eval_dataloader)

    gap = 10 / (config.train_iterator - 1)
    flag = True
    thres = 80
    while len(remain_sents) > 10000 and flag:
        thres = thres + random.randint(-3, 3)
        thres = min(thres, 83)
        thres = max(thres, 77)
        for i in range(config.train_iterator):
            train_sents, remain_sents = one_iter(bert, remain_sents, fq,
                                                 last_time_dataset=predicted_data,
                                                 thres=thres)
            if len(train_sents) < 32:
                print("Can only select %d sentences, iteration end." % len(train_sents))
                print("Remain %d sentences." % len(remain_sents))
                flag = False
                break

            eval_dataloader, _ = labeled_txt_to_dataloader(config, label_list, tokenizer, remain_sents)
            predicted_data = predict(bert_model, device, eval_dataloader)

    print("Second Part Done.")
    print("-*-" * 7)
    print("=&=" * 7)
    print("Remain %d sentences." % len(remain_sents))

    print("\nTest Part Start.")
    test_loader_list, label_list, book_list = load_data(config)
    for test_loader, book in zip(test_loader_list, book_list):
        print(book)
        test(bert_model, test_loader, config, device, n_gpu, label_list)
        sys.stdout.flush()
    print("Test Part Done.")

    # Third part is useless.
    '''
    # Third part, use bert to predict all the data and fine-tune bert.
    eval_dataloader, _ = labeled_txt_to_dataloader(config, label_list, tokenizer, raw_sents)
    predicted_data = predict(bert_model, device, eval_dataloader)
    sampler = SequentialSampler(predicted_data)
    train_dataloader = DataLoader(predicted_data, sampler=sampler, batch_size=config.train_batch_size)
    num_train_steps = int(len(train_dataloader.dataset) / config.train_batch_size * config.num_train_epochs)
    auto_train(bert_model, train_dataloader, config, device, n_gpu, label_list, num_train_steps, iters=3)
    print("Third Part Done.")
    print("-*-" * 7)
    print("=&=" * 7)
    '''


if __name__ == "__main__":
    '''
    args_config = args_parser()
    config = merge_config(args_config)
    test_loader_list, label_list, book_list = load_data(config)
    '''
    main()