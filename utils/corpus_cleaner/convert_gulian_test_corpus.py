
import os
import sys


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import re
import bs4
from bs4 import BeautifulSoup
from tqdm import tqdm


def main():
    with open('dataset/gulian_txt/比赛结果测试集.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    def get_line_combine(lines):
        lines = list(filter(None, lines))
        split_pattern = list("。！？」』")
        new_lines = []
        new_line = ''
        for line in lines:
            if len(line) == 1 and line[0] in split_pattern:
                new_line += line
            else:
                if new_line != '':
                    new_lines.append(new_line)
                    new_line = ''
                new_line += line
        if new_line != '':
            new_lines.append(new_line)
        return new_lines

    new_lines = []

    for line in lines:
        line = re.split(r'([。！？」』])', line.strip())
        line = get_line_combine(line)
        # line = [re.sub(pattern, '', l) for l in line]
        new_lines.extend(line)
        new_lines.append('')
    # new_lines = [i.strip() for i in lines]
    # new_lines = list(filter(None, lines))

    with open('dataset/gulian_txt/比赛结果测试集_new.txt', 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line + '\n')


if __name__ == '__main__':
    main()