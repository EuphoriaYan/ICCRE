#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import os 
import sys 

root_path = "/".join(os.path.realpath(__file__).split("/")[:-4])
if root_path not in sys.path:
    sys.path.append(root_path)


import math 
import numpy as np 


def cal_f1_score(pcs, rec):
    tmp = 2 * pcs * rec / (pcs + rec) 
    return round(tmp, 4) 


def extract_entities(labels_lst, start_label="1_4"):
    def gen_entities(label_lst, start_label, dims=1):
        # rules -> if end_mark > start_label 
        entities = dict()
        
        if "_" in start_label:
            start_label = start_label.split("_")
            start_label = [int(_) for _ in start_label]
            ind_func = lambda x: (bool(label in start_label) for label in x)
            indicator = sum([int(tmp) for tmp in ind_func(label_lst)])
        else:
            start_label = int(start_label)
            indicator = 1 if start_label in labels_lst else 0 

        if indicator > 0:
            if isinstance(start_label, list):
                ixs, _ = zip(*filter(lambda x: x[1] in start_label, enumerate(label_lst))) 
            elif isinstance(start_label, int):
                ixs, _ = zip(*filter(lambda x: x[1] == start_label, enumerate(label_lst)))
            else:
                raise ValueError("You Should Notice that The FORMAT of your INPUT")

            ixs = list(ixs)
            ixs.append(len(label_lst))
            for i in range(len(ixs) - 1):
                sub_label = label_lst[ixs[i]: ixs[i+1]]
                end_mark = max(sub_label)
                end_ix = ixs[i] + sub_label.index(end_mark) + 1 
                entities["{}_{}".format(ixs[i], end_ix)] = label_lst[ixs[i]: end_ix]
        return entities

    if start_label == "1":
        entities = gen_entities(labels_lst, start_label=start_label)
    elif start_label == "4":
        entities = gen_entities(labels_lst, start_label=start_label)
    elif "_" in start_label:
        entities = gen_entities(labels_lst, start_label=start_label)
    else:
        raise ValueError("You Should Check The FOMAT Of your SPLIT NUMBER")

    return entities 


def split_index(label_list):
    label_dict = {label: i for i, label in enumerate(label_list)}
    label_idx = [tmp_value for tmp_key, tmp_value in label_dict.items() if "S" in tmp_key.split("-")[0] or "B" in tmp_key]
    str_label_idx = [str(tmp) for tmp in label_idx]
    label_idx = "_".join(str_label_idx)
    return label_idx 


def compute_performance(pred_label, gold_label, pred_mask, label_list, dims=2, macro=False):
    start_label = split_index(label_list)

    if dims == 1:
        mask_index = [tmp_idx for tmp_idx, tmp in enumerate(pred_mask) if tmp != 0]
        pred_label = [tmp for tmp_idx, tmp in enumerate(pred_label) if tmp_idx in mask_index]
        gold_label = [tmp for tmp_idx, tmp in enumerate(gold_label) if tmp_idx in mask_index]

        pred_entities = extract_entities(pred_label, start_label = start_label)
        truth_entities = extract_entities(gold_label, start_label = start_label)

        # print("="*20)
        # print("pred entities")
        # print(pred_entities)
        # print(truth_entities)
        num_true = len(truth_entities)
        num_extraction = len(pred_entities)

        num_true_positive = 0 
        for entity_idx in pred_entities.keys():
            try:
                if truth_entities[entity_idx] == pred_entities[entity_idx]:
                    num_true_positive += 1 
            except:
                pass 

        dict_match = list(filter(lambda x: x[0] == x[1], zip(pred_label, gold_label)))
        acc = len(dict_match) / float(len(gold_label))

        if not macro:
            return acc, num_true_positive, float(num_extraction), float(num_true)

        if num_extraction != 0:
            pcs = num_true_positive / float(num_extraction)
        else:
            pcs = 0 

        if num_true != 0:
            recall = num_true_positive / float(num_true)
        else:
            recall = 0 

        if pcs + recall != 0 :
            f1 = 2 * pcs * recall / (pcs + recall)
        else:
            f1 = 0 

        if num_extraction == 0 and num_true == 0:
            acc, pcs, recall, f1 = 0, 0, 0, 0 
        acc, pcs, recall, f1 = round(acc, 4), round(pcs, 4), round(recall, 4), round(f1, 4)

        return acc, pcs, recall, f1 

    elif dims == 2:
        if not macro:
            acc, posit, extra, true = 0, 0, 0, 0
            for pred_item, truth_item, mask_item in zip(pred_label, gold_label, pred_mask):
                tmp_acc, tmp_posit, tmp_extra, tmp_true = compute_performance(pred_item, truth_item, mask_item, label_list, dims=1)
                posit += tmp_posit 
                extra += tmp_extra 
                true += tmp_true 
                acc += tmp_acc 

            if extra != 0:
                pcs = posit / float(extra)
            else:
                pcs = 0 

            if true != 0:
                recall = posit / float(true)
            else:
                recall = 0 

            if pcs + recall != 0 :
                f1 = 2 * pcs * recall / (pcs + recall)
            else:
                f1 = 0 
            acc = acc / len(pred_label)
            acc, pcs, recall, f1 = round(acc, 4), round(pcs, 4), round(recall, 4), round(f1, 4)
            return acc, pcs, recall, f1 

        acc_lst = []
        pcs_lst = []
        recall_lst = []
        f1_lst = []

        for pred_item, truth_item, mask_item in zip(pred_label, gold_label, pred_mask):
            tmp_acc, tmp_pcs, tmp_recall, tmp_f1 = compute_performance(pred_item, truth_item, \
                mask_item, label_list, dims=1, macro=True)
            if tmp_acc == 0.0 and tmp_pcs == 0 and tmp_recall == 0 and tmp_f1 == 0:
                continue 
            acc_lst.append(tmp_acc)
            pcs_lst.append(tmp_pcs)
            recall_lst.append(tmp_recall)
            f1_lst.append(tmp_f1)

        aveg_acc = round(sum(acc_lst)/len(acc_lst), 4)
        aveg_pcs = round(sum(pcs_lst)/len(pcs_lst), 4)
        aveg_recall = round(sum(recall_lst)/len(recall_lst), 4)
        aveg_f1 = round(sum(f1_lst)/len(f1_lst), 4)

        return aveg_acc, aveg_pcs, aveg_recall, aveg_f1 


if __name__ == "__main__":
    model_pred = [0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 3, 4, 1, 2, 1, 1]
    # entities = extract_entities(model_pred, start_label="1_3")
    # print(entities)

    label_list = ["O", "B-noun_bookname", "I-noun_bookname", "B-noun_other", "I-noun_other"]
    # label_idx = split_index(label_list)
    # print(label_idx)

    pred_list = [label_list[i] for i in model_pred]

    import itertools

    sent = ''
    idx = 0
    group_label = []
    for key, group in itertools.groupby(pred_list):
        l = len(list(group))
        if key.startswith('B') and l > 1:
            for i in range(l):
                group_label.append((key, 1))
        else:
            group_label.append((key, l))
    group_label_final = []
    for i in range(len(group_label)):
        key, l = group_label[i]
        # 是begin
        if key.startswith('B'):
            # 不是最后一个，可以往后看
            if i < len(group_label) - 1:
                nxt_key, nxt_l = group_label[i + 1]
                # 合并begin和inner为一组
                if nxt_key.startswith('I'):
                    group_label_final.append((key[2:], l + nxt_l))
                    i += 1
                # 后面一个不是I，说明是单独的B，单独一组
                else:
                    group_label_final.append((key[2:], l))
            # 最后一个，并且B开头，只能单独一组
            else:
                group_label_final.append((key[2:], l))
        elif key.startswith('I'):
            continue
        # 应该只剩O
        else:
            group_label_final.append((key, l))
    print(group_label)
    print(group_label_final)