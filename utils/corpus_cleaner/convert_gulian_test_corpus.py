#-*- coding:utf-8 -*-
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
from pprint import pprint

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
    split_pattern = list("。！？：；」』）’”")
    new_lines = []
    new_line = ''
    for line in lines:
        # “寔之政論言當世理亂，雖鼂錯之徒不能過也。”
        if line[0] in split_pattern:
            if len(line) == 1:
                new_line += line
            else:
                while len(line) > 0 and line[0] in split_pattern:
                    new_line += line[0]
                    line = line[1:]
                if new_line != '':
                    if len(new_line) <= MAX_SEQ_LEN:
                        new_lines.append(new_line)
                    else:
                        new_lines.extend(split_long_sent(new_line))
                    new_line = ''
                if line != '':
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


    with open('dataset/gulian_txt/命名实体识别测试集922_new_v0.4.txt', 'w', encoding='utf-8') as f:
        for line in new_lines:
            try:
                _ = int(line)
            except ValueError:
                assert len(line) <= MAX_SEQ_LEN, line
                f.write(line + '\n')


if __name__ == '__main__':
    line = '龐焕曰：「工者貴無與争，故大上用計謀，其次因人事，其下戰克。用計謀者，熒惑敵國之主，使變更淫俗，哆暴驕恣，而無聖人之數，愛人而與，無功而爵，未勞而賞，喜則釋罪，怒則妄殺，法民而自慎，少人而自至，繁無用，嗜龜占，󲳴󲳴高義，下合意内之人，所謂因人事者，結弊帛，用貨財，閉近人之復其口，使其所謂是者盡非也，所謂非者盡是也，離君之際用忠臣之路。所謂戰克者，其國已素破，兵從而攻之。因句踐用此而吴國亡，楚用此而陳蔡舉，三家用此而智氏亡，韓用此而東分。今世之言兵也，皆强大者必勝，小弱者必滅，是則小國之君無霸王者，而萬乘之主無破亡也。昔夏廣而湯狹，殷大而周小，越弱而吴强，此所謂不戰而勝，善之善者也。此陰經之法，夜行之道，天武之類也。今或僵尸百萬，流血千里，而勝未決也，以爲功計之，每已不若。是故聖人昭然獨思，忻然獨喜。若夫耳聞金鼓之聲而希功，目見旌旗之色而希陳，手握兵刃之枋而希戰，出進合闘而希勝，是襄主之所破亡也。」'
    line = re.split(r'([。：；！？」』])', line.strip())
    line = get_line_combine(line)
    pprint(line)
    l = '用計謀者，熒惑敵國之主，使變更淫俗，哆暴驕恣，而無聖人之數，愛人而與，無功而爵，未勞而賞，喜則釋罪，怒則妄殺，法民而自慎，少人而自至，繁無用，嗜龜占，\U000f2cf4\U000f2cf4高義，下合意内之人，所謂因人事者，結弊帛，用貨財，閉近人之復其口，使其所謂是者盡非也，'
    print(len(l))
    # main()