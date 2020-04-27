import re
import bs4
from bs4 import BeautifulSoup
import os
from tqdm import tqdm
from utils.tokenization import BasicTokenizer
import unicodedata

tokenizer = BasicTokenizer()


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def clean_str(text):
    global tokenizer
    tokenizer._clean_text(text)
    text = tokenizer._tokenize_chinese_chars(text)
    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
        token = tokenizer._run_strip_accents(token)
        split_tokens.extend(tokenizer._run_split_on_punc(token))
    return ''.join(split_tokens)


class dzg_reader:
    def read_dzg_file(self, data_dir):
        art_path = os.path.join(data_dir, 'art')
        art_file_list = os.listdir(art_path)
        art_train, art_dev = self.read_files(art_path, art_file_list, 'art')
        bud_path = os.path.join(data_dir, 'buddhism')
        bud_file_list = os.listdir(bud_path)
        bud_train, bud_dev = self.read_files(bud_path, bud_file_list, 'bud')
        med_path = os.path.join(data_dir, 'medicine')
        med_file_list = os.listdir(med_path)
        med_train, med_dev = self.read_files(med_path, med_file_list, 'med')
        tao_path = os.path.join(data_dir, 'taoism')
        tao_file_list = os.listdir(tao_path)
        tao_train, tao_dev = self.read_files(tao_path, tao_file_list, 'tao')
        train_list = []
        train_list.extend(art_train)
        train_list.extend(bud_train)
        train_list.extend(med_train)
        train_list.extend(tao_train)
        dev_list = []
        dev_list.extend(art_dev)
        dev_list.extend(bud_dev)
        dev_list.extend(med_dev)
        dev_list.extend(tao_dev)

        with open(os.path.join(data_dir, 'train.txt'), 'w', encoding='utf-8') as t:
            for line in train_list:
                text = line[0]
                label = line[1]
                t.write(text+'|'+label+'\n')

        with open(os.path.join(data_dir, 'dev.txt'), 'w', encoding='utf-8') as t:
            for line in dev_list:
                text = line[0]
                label = line[1]
                t.write(text+'|'+label+'\n')

    def read_files(self, data_dir, file_list, label):
        lines = []
        for file_name in file_list:
            with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    line = line.replace('|', '')
                    line = clean_str(line)
                    if len(line) > 4:
                        lines.append((line, label))
        l = len(lines)
        cut_point = l//8*7
        return lines[:cut_point], lines[cut_point:]


if __name__ == '__main__':
    path = 'dataset/clf/dzg'
    reader = dzg_reader()
    reader.read_dzg_file(path)


