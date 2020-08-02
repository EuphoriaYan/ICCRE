import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


def main():
    with open('output/gulian_test_ner_result.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open('output/gulian_test_ner_result_final.txt', 'w', encoding='utf-8') as f:
        for line in lines:
            line = line.strip()
            if line == '':
                f.write('\n')
            else:
                f.write(line)


if __name__ == '__main__':
    main()