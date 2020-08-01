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

from utils.tokenization import BertTokenizer, CompTokenizer
from utils.optimization import BertAdam, warmup_linear
from utils.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
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
    parser.add_argument("--data_dir", default=None, type=str, help="the input data dir")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="bert-large-uncased, bert-base-cased, bert-large-cased")
    parser.add_argument("--task_name", default=None, type=str)

    # # other parameters
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--use_multi_gpu", type=bool, default=False)
    parser.add_argument("--max_seq_length", default=128,
                        type=int, help="the maximum total input sequence length after ")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval")
    parser.add_argument("--use_comp", action="store_true")
    parser.add_argument("--use_crf", action="store_true")

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
    parser.add_argument("--export_model", type=bool, default=False)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--data_sign", type=str, default="msra_ner")
    parser.add_argument("--output_model_name", type=str, default="pytorch_model.bin")
    parser.add_argument("--ckpt_name", type=str, default="pytorch_model.bin")
    parser.add_argument("--output_file", action="store_true")
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


def load_book_data(config):
    # load some data and processor
    data_processor_list = {
        "msra_ner": MsraNERProcessor,
        "resume_ner": ResumeNERProcessor,
        "ontonotes_ner": OntoNotesNERProcessor,
        "ctb5_pos": Ctb5POSProcessor,
        "ctb6_pos": Ctb6CWSProcessor,
        "ud1_pos": Ud1POSProcessor,
        "ctb6_cws": Ctb6CWSProcessor,
        "pku_cws": PkuCWSProcessor,
        "msr_cws": MsrCWSProcessor,
        "zuozhuan_cws": ZuozhuanCWSProcessor,
        "book_cws": BookCWSProcessor,
        "artical_cws": ArticalCWSProcessor,
        "artical_ner": ArticalNERProcessor,
        "gulian_ner": GLNEWNERProcessor,
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
    test_examples_list, book_list = data_processor.get_test_examples(config.data_dir)

    def generate_data(examples, sampler_method='random'):
        features = convert_examples_to_features(examples, label_list, config.max_seq_length, tokenizer,
                                                task_sign=config.task_name)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        char_mask = torch.tensor([f.char_mask for f in features], dtype=torch.bool)
        label_len = torch.tensor([f.label_len for f in features], dtype=torch.long)
        return input_ids, input_mask, segment_ids, label_ids, char_mask, label_len
        # data = TensorDataset(input_ids, input_mask, segment_ids, label_ids, label_len)
        # sampler = DistributedSampler(data)
        # sampler_list = {
        #    'random': RandomSampler,
        #    'sequential': SequentialSampler,
        # }
        # if sampler_method not in sampler_list:
        #    raise ValueError("sample method not found.")
        # sampler = sampler_list[sampler_method](data)
        # return data, sampler

    # convert data example into featrues

    test_data_list = []
    # test_sampler_list = []
    for test_examples in test_examples_list:
        input_ids, input_mask, segment_ids, label_ids, char_mask, label_len = generate_data(test_examples)
        # test_data, test_sampler = generate_data(test_examples, 'sequential')
        test_data_list.append([input_ids, input_mask, segment_ids, label_ids, char_mask, label_len])
        # test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=config.test_batch_size)
        # test_dataloader_list.append(test_dataloader)

    return test_data_list, label_list, book_list, tokenizer


def load_data(config):
    # load some data and processor
    data_processor_list = {
        "msra_ner": MsraNERProcessor,
        "resume_ner": ResumeNERProcessor,
        "ontonotes_ner": OntoNotesNERProcessor,
        "ctb5_pos": Ctb5POSProcessor,
        "ctb6_pos": Ctb6CWSProcessor,
        "ud1_pos": Ud1POSProcessor,
        "ctb6_cws": Ctb6CWSProcessor,
        "pku_cws": PkuCWSProcessor,
        "msr_cws": MsrCWSProcessor,
        "zuozhuan_cws": ZuozhuanCWSProcessor,
        "book_cws": BookCWSProcessor,
        "artical_cws": ArticalCWSProcessor,
        "artical_ner": ArticalNERProcessor,
        "gulian_ner": GLNEWNERProcessor,
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
    test_examples = data_processor.get_test_examples(config.data_dir)

    def generate_data(examples, sampler_method='random'):
        features = convert_examples_to_features(examples, label_list, config.max_seq_length, tokenizer,
                                                task_sign=config.task_name)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        char_mask = torch.tensor([f.char_mask for f in features], dtype=torch.bool)
        label_len = torch.tensor([f.label_len for f in features], dtype=torch.long)
        return input_ids, input_mask, segment_ids, label_ids, char_mask, label_len
        # data = TensorDataset(input_ids, input_mask, segment_ids, label_ids, label_len)
        # sampler = DistributedSampler(data)
        # sampler_list = {
        #    'random': RandomSampler,
        #    'sequential': SequentialSampler,
        # }
        # if sampler_method not in sampler_list:
        #    raise ValueError("sample method not found.")
        # sampler = sampler_list[sampler_method](data)
        # return data, sampler

    # convert data example into featrues

    input_ids, input_mask, segment_ids, label_ids, char_mask, label_len = generate_data(test_examples)
    # test_data, test_sampler = generate_data(test_examples, 'sequential')
    test_data = [input_ids, input_mask, segment_ids, label_ids, char_mask, label_len]
    # test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=config.test_batch_size)
    # test_dataloader_list.append(test_dataloader)

    return test_data, label_list, tokenizer


def load_model(config, label_list):
    # device = torch.device(torch.cuda.is_available())

    if torch.cuda.is_available():
        if config.use_multi_gpu:
            n_gpu = torch.cuda.device_count()
            # device = torch.device("cuda")
        else:
            n_gpu = 1
            # device = torch.device(config.device)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        n_gpu = 0

    model = BertTagger(config, num_labels=len(label_list))
    model_path = os.path.join(config.output_dir, config.ckpt_name)
    if n_gpu > 0:
        model.load_state_dict(torch.load(model_path, map_location='cuda'))
        model.to(device)
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model, device, n_gpu


def convert_feature_to_sents(dataset, tokenizer, label_list, task_name):
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

        if task_name == 'BIO_cws':
            sent = ''
            for t, l in zip(raw_tokens, raw_label):
                if l == 'B':
                    sent += ' '
                    sent += t
                else:
                    sent += t
            sent = sent.strip()
            sents.append(sent)
        elif task_name == 'ner':
            import itertools
            sent = ''
            idx = 0
            for key, group in itertools.groupby(raw_label):
                l = len(list(group))
                for i in range(l):
                    sent += raw_tokens[idx]
                    idx += 1
                if key == 'O':
                    sent += ' '
                else:
                    sent += '/' + key + ' '
            sents.append(sent)

    return sents


def test(model, test_dataloader, config, device, n_gpu, label_list, tokenizer):
    if not config.output_file:
        test_loss, test_prec, test_rec, test_f1 = eval_checkpoint(
            model, test_dataloader, device, label_list,
            config.task_name, config.use_crf)
        print("TEST: precision, recall, f1, loss ")
        print(test_prec, test_rec, test_f1, test_loss, '\n')
        return 0
    else:
        logits = eval_checkpoint(model, test_dataloader, device, label_list,
            config.task_name, config.use_crf, config.output_file)
        logits = torch.tensor(logits, dtype=torch.long)
        return logits


def merge_config(args_config):
    model_config_path = args_config.config_path
    model_config = Config.from_json_file(model_config_path)
    model_config.update_args(args_config)
    model_config.print_config()
    return model_config


def article_ckpt():
    args_config = args_parser()
    config = merge_config(args_config)
    test_data_list, label_list, book_list, tokenizer = load_book_data(config)
    model, device, n_gpu = load_model(config, label_list)
    for test_data, book in zip(test_data_list, book_list):
        print(book)
        test_dataset = TensorDataset(*test_data)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=config.test_batch_size)
        output = test(model, test_dataloader, config, device, n_gpu, label_list, tokenizer)
        if output is 0:
            continue
        else:
            input_ids = test_data[0]
            input_mask = test_data[1]
            segment_ids = test_data[2]
            label_ids = output
            label_len = test_data[4]
            new_dataset = TensorDataset(input_ids, input_mask, segment_ids, label_ids, label_len)
            sents = convert_feature_to_sents(new_dataset, tokenizer, label_list, config.task_name)
            with open(os.path.join(config.output_dir, book), 'w', encoding='utf-8') as f:
                for i in sents:
                    f.write(i+'\n')
        sys.stdout.flush()


def main():
    args_config = args_parser()
    config = merge_config(args_config)
    test_data, label_list, tokenizer = load_data(config)
    model, device, n_gpu = load_model(config, label_list)
    test_dataset = TensorDataset(*test_data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=config.test_batch_size)
    output = test(model, test_dataloader, config, device, n_gpu, label_list, tokenizer)
    if output is not 0:
        input_ids = test_data[0]
        input_mask = test_data[1]
        segment_ids = test_data[2]
        label_ids = output
        label_len = test_data[4]
        new_dataset = TensorDataset(input_ids, input_mask, segment_ids, label_ids, label_len)
        sents = convert_feature_to_sents(new_dataset, tokenizer, label_list, config.task_name)
        with open(os.path.join(config.output_dir, config.data_sign + '_result.txt'), 'w', encoding='utf-8') as f:
            for i in sents:
                f.write(i + '\n')



if __name__ == "__main__":
    main()
