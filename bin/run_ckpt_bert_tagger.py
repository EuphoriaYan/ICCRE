#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

server_root_path = ''

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
    parser.add_argument("--use_server", action="store_true")
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
        "book_cws": BookCWSProcessor
    }

    if config.data_sign not in data_processor_list:
        raise ValueError("Data_sign not found: %s" % config.data_sign)

    data_processor = data_processor_list[config.data_sign]()

    label_list = data_processor.get_labels()
    if 'wcm' in config.ckpt_name:
        tokenizer = CompTokenizer(config.bert_model)
    else:
        tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)

    # load data exampels
    test_examples_list, book_list = data_processor.get_test_examples(config.data_dir)

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


def load_model(config, label_list):
    # device = torch.device(torch.cuda.is_available())

    if config.use_server and torch.cuda.is_available():
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
    if config.use_server:
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    return model, device, n_gpu


def test(model, test_dataloader, config, device, n_gpu, label_list):
    test_loss, test_acc, test_prec, test_rec, test_f1 = eval_checkpoint(
        model_object=model,
        eval_dataloader=test_dataloader,
        device=device,
        label_list=label_list,
        task_sign=config.task_name,
        use_crf=config.use_crf
    )

    print("TEST: precision, recall, f1, acc, loss ")
    print(test_prec, test_rec, test_f1, test_acc, test_loss, '\n')
    return


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


def main():
    args_config = args_parser()
    config = merge_config(args_config)
    test_loader_list, label_list, book_list = load_data(config)
    model, device, n_gpu = load_model(config, label_list)
    for test_loader, book in zip(test_loader_list, book_list):
        print(book)
        test(model, test_loader, config, device, n_gpu, label_list)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
