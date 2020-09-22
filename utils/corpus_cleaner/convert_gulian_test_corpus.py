
import os
import sys


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import re
import bs4
from bs4 import BeautifulSoup
from tqdm import tqdm
import textwrap

MAX_SEQ_LEN=126


def split_long_sent(line):
    new_lines = []
    new_line = ''
    pieces = re.split(r'([，。！？：、‘’“”])', line.strip())
    for piece in pieces:
        if new_line != '' and len(new_line) + len(piece) > MAX_SEQ_LEN:  # 超出长度，塞到list里
            while len(new_line) > MAX_SEQ_LEN:
                new_lines.append(new_line[:MAX_SEQ_LEN])
                new_line = new_line[MAX_SEQ_LEN:]
            if new_line != '':
                new_lines.append(new_line)
            new_line = ''
        else:
            new_line += piece

    if new_line != '':
        while len(new_line) > MAX_SEQ_LEN:
            new_lines.append(new_line[:MAX_SEQ_LEN])
            new_line = new_line[MAX_SEQ_LEN:]
        if new_line != '':
            new_lines.append(new_line)

    return new_lines


def get_line_combine(lines):
    lines = list(filter(None, lines))
    split_pattern = list("。！？：；」』’”")
    new_lines = []
    new_line = ''
    for line in lines:
        # “寔之政論言當世理亂，雖鼂錯之徒不能過也。”
        if len(line) == 1 and line[0] in split_pattern:
            new_line += line
        else:
            if new_line != '':
                if len(new_line) <= MAX_SEQ_LEN:
                    new_lines.append(new_line)
                else:
                    new_lines.extend(split_long_sent(new_line))
                new_line = ''
            new_line += line
    if new_line != '':
        if len(new_line) <= MAX_SEQ_LEN:
            new_lines.append(new_line)
        else:
            new_lines.extend(split_long_sent(new_line))
    return new_lines


def main():
    with open('dataset/gulian_txt/命名实体识别测试集922.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        line = re.split(r'([。：；！？」』])', line.strip())
        line = get_line_combine(line)
        # line = [re.sub(pattern, '', l) for l in line]
        new_lines.extend(line)
        new_lines.append('')
    # new_lines = [i.strip() for i in lines]
    # new_lines = list(filter(None, lines))


    with open('dataset/gulian_txt/命名实体识别测试集922_new.txt', 'w', encoding='utf-8') as f:
        for line in new_lines:
            assert len(line) <= MAX_SEQ_LEN, line
            f.write(line + '\n')


if __name__ == '__main__':
    line = '“寔之政論言當世理亂，雖鼂錯之徒不能過也。”'
    line = re.split(r'([。：；！？」』])', line.strip())
    get_line_combine(line)