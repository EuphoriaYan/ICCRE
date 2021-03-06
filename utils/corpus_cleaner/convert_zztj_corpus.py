
import os
import sys


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import re
import bs4
from bs4 import BeautifulSoup
from tqdm import tqdm

from utils.tokenization import BasicTokenizer

tokenizer = BasicTokenizer()


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def clean_str(text):
    global tokenizer
    tokenizer._clean_text(text)
    text = tokenizer._tokenize_chinese_chars(text)
    orig_tokens = whitespace_tokenize(text)
    return ''.join(orig_tokens)


class zztj_reader():
    def read_zztj_file(self, file_name):

        # pre_clean
        '''
        def analyse(tagged_html):
            l = []
            ano_l = []
            for child in tagged_html.children:
                if type(child) == bs4.Tag and child['class'][0] == 'ano':
                    ano_l.extend(analyse(child))
                else:
                    l.append(str(child).strip())
            l.extend(ano_l)
            return l

        with open(os.path.join('dataset/ner/zztj', file_name), 'r', encoding='utf-8') as f:
            raw_html = f.readline().strip().strip('"')
        raw_html = raw_html.replace(r'\/', '/').replace(r'\"', '"')
        soup = BeautifulSoup(raw_html, 'lxml')
        soup = soup.p
        components = analyse(soup)
        components = list(filter(None, components))
        processed_html = ''.join(components)
        processed_html = processed_html.replace('【', '').replace('】', '').replace(r'\n', '\n')
        processed_html = processed_html.split('\n')
        processed_html = list(filter(None, processed_html))
        pattern = re.compile('^\d+')
        processed_html = [re.sub(pattern, '', l.strip()) for l in processed_html]

        with open(os.path.join('dataset/ner/zztj', file_name), 'w', encoding='utf-8') as f:
            for line in processed_html:
                f.write(line.strip() + '\n')
        '''

        global tokenizer
        with open(os.path.join('dataset/ner/zztj', file_name), 'r', encoding='utf-8') as f:
            processed_html = f.readlines()

        # processed_html = [re.sub('[「」『』、]', '', l.strip()) for l in processed_html]
        processed_html = [l.replace('○', '零')
                              .replace('「', '“')
                              .replace('」', '”')
                              .replace('『', '‘')
                              .replace('』', '’') for l in processed_html]
        pattern = re.compile('^\d+')
        processed_html = [re.sub(pattern, '', l.strip()) for l in processed_html]
        lines = []

        # split_pattern = re.compile(r'[，。：；！？]')
        # split_pattern = re.compile()

        def get_line_combine(lines):
            lines = list(filter(None, lines))
            split_pattern = list("。！？”’")
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

        for line in processed_html:
            line = re.split(r'([。！？”’])', line)
            line = get_line_combine(line)
            # line = [re.sub(pattern, '', l) for l in line]
            lines.extend(line)
        lines = [i.strip() for i in lines]
        lines = list(filter(None, lines))

        res = []
        for raw_line in lines:
            soup = BeautifulSoup(raw_line, 'lxml')
            soup = soup.body
            if soup is None:
                print(file_name)
                print(raw_line)
                continue
            if soup.p is not None:
                soup = soup.p
            line = ''
            label = []
            for child in soup.children:
                if type(child) == bs4.Tag:
                    cls = child['class'][0]
                    s = str(child.text)
                    s = clean_str(s)
                    line += s
                    for i in range(len(s)):
                        if i == 0:
                            label.append('B-' + cls)
                        else:
                            label.append('I-' + cls)
                else:
                    s = str(child)
                    s = clean_str(s)
                    line += s
                    for i in range(len(s)):
                        label.append('O')

            assert len(line) == len(label)
            res.append((line, label))
        return res


if __name__ == '__main__':
    reader = zztj_reader()
    threads = []
    for i in tqdm(range(1, 295)):
        text_name = str(i) + '.txt'
        lines = reader.read_zztj_file(text_name)
        with open(os.path.join('dataset/ner/zztj_new', text_name), 'w', encoding='utf-8') as f:
            for line in lines:
                text, label = line
                f.write(text + '|' + ' '.join(label) + '\n')
