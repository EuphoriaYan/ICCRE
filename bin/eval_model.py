#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def eval_checkpoint(model_object, eval_dataloader, device, label_list, task_sign='ner', use_crf=False,
                    output_file=False):
    # input_dataloader type can only be one of dev_dataloader, test_dataloader
    model_object.eval()

    idx2label = {i: label for i, label in enumerate(label_list)}

    loss = 0
    pred_lst = []
    gold_lst = []
    lst_len = []
    eval_steps = 0

    for batch in eval_dataloader:
        batch = (t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, char_mask, label_len = batch

        with torch.no_grad():
            if not output_file:
                tmp_eval_loss = model_object(input_ids, segment_ids, input_mask, label_ids, char_mask, use_crf)
            logits = model_object(input_ids, segment_ids, input_mask,
                                  labels=None, char_mask=char_mask, use_crf=use_crf)

        """
        logits = logits.detach().cpu().numpy()
        logits = np.argmax(logits, axis=-1)
        label_ids = label_ids.to("cpu").numpy()
        label_len = label_len.to("cpu").numpy()
        reshape_lst = label_ids.shape
        logits = np.reshape(logits, (reshape_lst[0], reshape_lst[1], -1))
        logits = np.argmax(logits, axis=-1)
        logits = logits.tolist()
        """

        if not output_file:
            char_mask = char_mask.view(-1, input_ids.shape[1])
            char_mask = char_mask.cpu().numpy()  # (batch_size, seq_length)
            label_ids = label_ids.cpu().numpy()
            labels = []
            for i in range(char_mask.shape[0]):
                l = label_ids[i]
                c = char_mask[i]
                l = l[c]
                l = l.tolist()
                labels.append(l)

            label_len = label_len.tolist()

            loss += tmp_eval_loss.mean().item()

            pred_lst += logits
            gold_lst += labels
            lst_len += label_len
            eval_steps += 1
        else:
            pred_lst += logits

    if output_file:
        return pred_lst

    average_loss = round(loss / eval_steps, 4)

    if task_sign == 'BIO_cws':
        p = 0
        tot_p = 0
        r = 0
        tot_r = 0
        for pred, gold, l in zip(pred_lst, gold_lst, lst_len):
            for pd, gd in zip(pred[1:], gold[1:]):
                if pd == 0:
                    tot_p += 1
                    if gd == 0:
                        p += 1
                if gd == 0:
                    tot_r += 1
                    if pd == 0:
                        r += 1
        precision = p / tot_p
        recall = r / tot_r
        f1 = 2 * precision * recall / (precision + recall)
        eval_f1 = round(f1, 4)
        eval_precision = round(precision, 4)
        eval_recall = round(recall, 4)
        return average_loss, eval_precision, eval_recall, eval_f1

    elif task_sign == 'ner':
        pred_all = []
        gold_all = []
        for pred, gold, l in zip(pred_lst, gold_lst, lst_len):
            p_temp = pred[1:l + 1]
            g_temp = gold[1:l + 1]
            if len(p_temp) != len(g_temp):
                print(p_temp)
                print(g_temp)
                print(l)
                continue
            pred_all.extend(p_temp)
            gold_all.extend(g_temp)

        precision = precision_score(gold_all, pred_all, average='macro')
        recall = recall_score(gold_all, pred_all, average='macro')
        f1 = f1_score(gold_all, pred_all, average='macro')
        eval_f1 = round(f1, 4)
        eval_precision = round(precision, 4)
        eval_recall = round(recall, 4)
        return average_loss, eval_precision, eval_recall, eval_f1

    elif task_sign == 'cws+pos':
        cws_pred = []
        cws_gold = []
        ner_pred = []
        ner_gold = []
        for pred, gold, l in zip(pred_lst, gold_lst, lst_len):
            pred = pred[1:l + 1]
            gold = gold[1:l + 1]
            pred = [idx2label[p].split('-') for p in pred]
            gold = [idx2label[g].split('-') for g in gold]
            cws_pred.extend([p[0] for p in pred])
            cws_gold.extend([g[0] for g in gold])
            ner_pred.extend([p[1] for p in pred])
            ner_gold.extend([g[1] for g in gold])

        c_p = precision_score(cws_gold, cws_pred, average='binary', pos_label='B')
        c_r = recall_score(cws_gold, cws_pred, average='binary', pos_label='B')
        c_f1 = 2 * (c_p * c_r) / (c_p + c_r)
        n_p = precision_score(ner_gold, ner_pred, average='macro')
        n_r = recall_score(ner_gold, ner_pred, average='macro')
        n_f1 = 2 * (n_p * n_r) / (n_p + n_r)
        c_f1, c_p, c_r = round(c_f1, 4), round(c_p, 4), round(c_r, 4)
        n_f1, n_p, n_r = round(n_f1, 4), round(n_p, 4), round(n_r, 4)
        return average_loss, c_p, c_r, c_f1, n_p, n_r, n_f1

    elif task_sign == 'pos':
        pos_pred = []
        pos_gold = []
        for pred, gold, l in zip(pred_lst, gold_lst, lst_len):
            pred = pred[1:l + 1]
            gold = gold[1:l + 1]
            pos_pred.extend(pred)
            pos_gold.extend(gold)

        p_p = precision_score(pos_gold, pos_pred, average='macro')
        p_r = recall_score(pos_gold, pos_pred, average='macro')
        p_f1 = 2 * (p_p * p_r) / (p_p + p_r)
        return average_loss, p_p, p_r, p_f1


def convert2entity(tag_lst, task_sign):
    res_lst = []
    if task_sign == 'BIO_cws':
        last = 0
        for i in reversed(range(len(tag_lst))):
            if tag_lst[i] == 'I' and last == 0:
                last = i
            if tag_lst[i] == 'B':
                if last != 0:
                    res_lst.append(str(i) + '_' + str(last))
                else:
                    res_lst.append(str(i))
                last = 0
    if task_sign == 'ner':
        start = -1
        for i in range(len(tag_lst)):
            if tag_lst[i].startswith('S'):
                res_lst.append(str(i))
            elif tag_lst[i].startswith('B'):
                start = i
            elif tag_lst[i].startswith('E'):
                if start == -1:
                    raise ValueError("E before B")
                start = -1
                res_lst.append(str(start) + '_' + str(i))
    return res_lst
