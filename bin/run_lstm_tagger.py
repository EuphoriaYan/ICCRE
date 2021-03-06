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
from utils.optimization import BertAdam, warmup_linear
from utils.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from utils.config import Config
from dataset_readers.bert_ner import *
from dataset_readers.bert_pos import *
from dataset_readers.bert_cws import *
from dataset_readers.bert_css import *
from models.BiLSTM_CRF.bilstm_crf import BiLSTM_CRF
from dataset_readers.bert_data_utils import convert_examples_to_features
from bin.train_model import train

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
    parser.add_argument("--export_model", type=bool, default=False)
    parser.add_argument("--output_model_name", type=str, default="pytorch_model.bin")
    parser.add_argument("--use_server", action="store_true")

    # # other parameters
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--use_multi_gpu", type=bool, default=False)
    parser.add_argument("--max_seq_length", default=128,
                        type=int, help="the maximum total input sequence length")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval")

    parser.add_argument("--embedding_size", default=192, type=int)
    parser.add_argument("--hidden_size", default=192, type=int)
    parser.add_argument("--lstm_dropout", default=0.2, type=float)
    parser.add_argument("--num_layers", default=2, type=int)

    parser.add_argument("--clip_grad", default=1.0, type=float)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--checkpoint", default=500, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--num_train_epochs", default=4.0, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--use_crf", type=bool, default=True)

    args = parser.parse_args()

    args.batch_size = args.batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError
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
        "ud1_pos": Ud1POSProcessor,
        "ctb6_cws": Ctb6CWSProcessor,
        "pku_cws": PkuCWSProcessor,
        "msr_cws": MsrCWSProcessor,
        "zuozhuan_cws": ZuozhuanCWSProcessor,
        "whitespace_cws": WhitespaceCWSPrecessor,
        "shiji_css": ShijiCSSProcessor,
        "sinica_ner": SinicaNERProcessor,
        "zuozhuan_pos": ZuozhuanPOSProcessor,
        "zztj_ner": ZztjNERProcessor,
    }

    if config.data_sign not in data_processor_list:
        raise ValueError("Data_sign not found: %s" % config.data_sign)

    data_processor = data_processor_list[config.data_sign]()

    label_list = data_processor.get_labels()
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

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.batch_size)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=config.batch_size)

    num_train_steps = int(math.ceil(len(train_examples) / config.batch_size) * config.num_train_epochs)
    print("Train_examples: ", len(train_examples))
    print("Dev_examples: ", len(dev_examples))
    print("Test_examples: ", len(test_examples))
    print("Total train steps: ", num_train_steps)
    return train_dataloader, dev_dataloader, test_dataloader, num_train_steps, label_list


def load_model(config, num_train_steps, label_list):
    # device = torch.device(torch.cuda.is_available())

    if torch.cuda.is_available():
        if config.use_multi_gpu:
            n_gpu = torch.cuda.device_count()
        else:
            n_gpu = 1
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        n_gpu = 0

    model = BiLSTM_CRF(config, len(label_list))
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = Adam(optimizer_grouped_parameters,
                     lr=config.learning_rate)

    '''optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=config.warmup_proportion,
                         t_total=num_train_steps,
                         max_grad_norm=config.clip_grad)
    '''

    return model, optimizer, device, n_gpu


def merge_config(args_config):
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
