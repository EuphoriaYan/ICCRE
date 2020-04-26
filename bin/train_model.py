#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
import time
from bin.eval_model import eval_checkpoint


def train(model, optimizer, train_dataloader, dev_dataloader, test_dataloader,
          config, device, n_gpu, label_list):
    train_list = {
        "BIO_cws": cws_train,
        "cws": cws_train,
        "ner": ner_train,
        "pos": pos_train,
        "cws+pos": cws_pos_train,
    }
    if config.task_name not in train_list:
        raise ValueError("task_name %s not found." % config.task_name)

    train_list[config.task_name](model, optimizer,
                                 train_dataloader, dev_dataloader, test_dataloader,
                                 config, device, n_gpu, label_list)


def cws_train(model, optimizer, train_dataloader, dev_dataloader, test_dataloader,
          config, device, n_gpu, label_list):

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    dev_best_precision = 0
    dev_best_recall = 0
    dev_best_f1 = 0
    dev_best_loss = 1e10

    test_best_precision = 0
    test_best_recall = 0
    test_best_f1 = 0
    test_best_loss = 1e10
    model.train()
    train_start = time.time()
    for idx in range(int(config.num_train_epochs)):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        print("#######" * 7)
        epoch_start = time.time()
        print("EPOCH: %s/%s" % (str(idx + 1), config.num_train_epochs))

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, char_mask, label_len = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids, char_mask, config.use_crf)
            if n_gpu > 1:
                loss = loss.mean()

            model.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)

            tr_loss += loss.item()

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                # optimizer.zero_grad()
                global_step += 1

            if (global_step + 1) % config.checkpoint == 0:
                print("-*-" * 15)
                print("current training loss is : ")
                print(loss.item())

                if config.do_eval:
                    dev_loss, dev_prec, dev_rec, dev_f1 = eval_checkpoint(model, dev_dataloader, device,
                                                                          label_list, config.task_name,
                                                                          use_crf=config.use_crf)

                    print("..." * 10)
                    print("checkpoint: " + str(int((global_step + 1) / config.checkpoint)))
                    print("DEV: loss,  precision, recall, f1")
                    print(dev_loss, dev_prec, dev_rec, dev_f1)

                    if dev_f1 > dev_best_f1:
                        dev_best_loss = dev_loss
                        dev_best_precision = dev_prec
                        dev_best_recall = dev_rec
                        dev_best_f1 = dev_f1
                        test_loss, test_prec, test_rec, test_f1 = eval_checkpoint(model, test_dataloader, device,
                                                                                            label_list, config.task_name,
                                                                                            use_crf=config.use_crf)
                        print("......" * 10)
                        print("TEST: loss, precision, recall, f1")
                        print(test_loss, test_prec, test_rec, test_f1)

                        if test_f1 > test_best_f1:
                            test_best_loss = test_loss
                            test_best_precision = test_prec
                            test_best_recall = test_rec
                            test_best_f1 = test_f1

                            if config.export_model:
                                # export a better model
                                model_to_save = model.module if hasattr(model, "module") else model
                                output_model_file = os.path.join(config.output_dir, config.output_model_name)
                                torch.save(model_to_save.state_dict(), output_model_file)
                            # end of if do_eval

                print("-*-" * 15, flush=True)
                model.train()
                # end of checkpoint

        epoch_finish = time.time()
        print("EPOCH: %d; TIME: %.2fs" % (idx, epoch_finish - epoch_start), flush=True)
        # end of epoch

    train_finish = time.time()
    print("TOTAL_TIME: %.2fs" % (train_finish - train_start))

    print("=&=" * 15)
    print("DEV: current best loss, p, r, f1 ")
    print(dev_best_loss, dev_best_precision, dev_best_recall, dev_best_f1)
    print("TEST: current best loss, p, r, f1 ")
    print(test_best_loss, test_best_precision, test_best_recall, test_best_f1)
    print("=&=" * 15)


def ner_train(model, optimizer, train_dataloader, dev_dataloader, test_dataloader,
          config, device, n_gpu, label_list):

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    dev_best_precision = 0
    dev_best_recall = 0
    dev_best_f1 = 0
    dev_best_loss = 1e10

    test_best_precision = 0
    test_best_recall = 0
    test_best_f1 = 0
    test_best_loss = 1e10

    model.train()
    train_start = time.time()
    for idx in range(int(config.num_train_epochs)):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        print("#######" * 7)
        epoch_start = time.time()
        print("EPOCH: %s/%s" % (str(idx + 1), config.num_train_epochs))

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, char_mask, label_len = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids, char_mask, config.use_crf)
            if n_gpu > 1:
                loss = loss.mean()

            model.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)

            tr_loss += loss.item()

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                # optimizer.zero_grad()
                global_step += 1

            if (global_step + 1) % config.checkpoint == 0:
                print("-*-" * 15)
                print("current training loss is : ")
                print(loss.item())

                if config.do_eval:
                    dev_loss, dev_prec, dev_rec, dev_f1 = eval_checkpoint(model, dev_dataloader, device,
                                                                          label_list, config.task_name,
                                                                          use_crf=config.use_crf)

                    print("..." * 10)
                    print("checkpoint: " + str(int((global_step + 1) / config.checkpoint)))
                    print("DEV: loss, acc, precision, recall, f1")
                    print(dev_loss, dev_prec, dev_rec, dev_f1)

                    if dev_f1 > dev_best_f1:
                        dev_best_loss = dev_loss
                        dev_best_precision = dev_prec
                        dev_best_recall = dev_rec
                        dev_best_f1 = dev_f1

                        if config.export_model:
                            # export a better model
                            model_to_save = model.module if hasattr(model, "module") else model
                            output_model_file = os.path.join(config.output_dir, config.output_model_name)
                            torch.save(model_to_save.state_dict(), output_model_file)
                # end of if do_eval

                print("-*-" * 15, flush=True)
                model.train()
                # end of checkpoint

        epoch_finish = time.time()
        print("EPOCH: %d; TIME: %.2fs" % (idx, epoch_finish - epoch_start), flush=True)
        # end of epoch

    train_finish = time.time()
    print("TOTAL_TIME: %.2fs" % (train_finish - train_start))

    print("=&=" * 15)
    print("DEV: current best loss, acc, p, r, f1 ")
    print(dev_best_loss, dev_best_precision, dev_best_recall, dev_best_f1)
    print("=&=" * 15)


def pos_train(model, optimizer, train_dataloader, dev_dataloader, test_dataloader,
          config, device, n_gpu, label_list):

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    dev_best_loss = 1e10
    dev_best_precision = 0
    dev_best_recall = 0
    dev_best_f1 = 0

    model.train()
    train_start = time.time()
    for idx in range(int(config.num_train_epochs)):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        print("#######" * 7)
        epoch_start = time.time()
        print("EPOCH: %s/%s" % (str(idx + 1), config.num_train_epochs))

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, char_mask, label_len = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids, char_mask, config.use_crf)
            if n_gpu > 1:
                loss = loss.mean()

            model.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)

            tr_loss += loss.item()

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                # optimizer.zero_grad()
                global_step += 1

            if (global_step + 1) % config.checkpoint == 0:
                print("-*-" * 15)
                print("current training loss is : ")
                print(loss.item())

                if config.do_eval:
                    dev_loss, dev_prec, dev_rec, dev_f1 = eval_checkpoint(model, dev_dataloader, device,
                                                                          label_list, config.task_name,
                                                                          use_crf=config.use_crf)

                    print("..." * 10)
                    print("checkpoint: " + str(int((global_step + 1) / config.checkpoint)))
                    print("DEV: loss, acc, precision, recall, f1")
                    print(dev_loss, dev_prec, dev_rec, dev_f1)

                    if dev_f1 > dev_best_f1:
                        dev_best_loss = dev_loss
                        dev_best_precision = dev_prec
                        dev_best_recall = dev_rec
                        dev_best_f1 = dev_f1

                        if config.export_model:
                            # export a better model
                            model_to_save = model.module if hasattr(model, "module") else model
                            output_model_file = os.path.join(config.output_dir, config.output_model_name)
                            torch.save(model_to_save.state_dict(), output_model_file)
                        # end of if do_eval

                print("-*-" * 15, flush=True)
                model.train()
                # end of checkpoint

        epoch_finish = time.time()
        print("EPOCH: %d; TIME: %.2fs" % (idx, epoch_finish - epoch_start), flush=True)
        # end of epoch

    train_finish = time.time()
    print("TOTAL_TIME: %.2fs" % (train_finish - train_start))

    print("=&=" * 15)
    print("DEV: current best loss, acc, p, r, f1 ")
    print(dev_best_loss, dev_best_precision, dev_best_recall, dev_best_f1)
    print("=&=" * 15)


def cws_pos_train(model, optimizer, train_dataloader, dev_dataloader, test_dataloader,
          config, device, n_gpu, label_list):

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    best_loss = 1e10
    cws_best_precision = 0
    cws_best_recall = 0
    cws_best_f1 = 0

    pos_best_precision = 0
    pos_best_recall = 0
    pos_best_f1 = 0

    model.train()
    train_start = time.time()
    for idx in range(int(config.num_train_epochs)):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        print("#######" * 7)
        epoch_start = time.time()
        print("EPOCH: %s/%s" % (str(idx + 1), config.num_train_epochs))

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, char_mask, label_len = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids, char_mask, config.use_crf)
            if n_gpu > 1:
                loss = loss.mean()

            model.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)

            tr_loss += loss.item()

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                # optimizer.zero_grad()
                global_step += 1

            if (global_step + 1) % config.checkpoint == 0:
                print("-*-" * 15)
                print("current training loss is : ")
                print(loss.item())

                if config.do_eval:
                    loss, c_p, c_r, c_f1, p_p, p_r, p_f1 = eval_checkpoint(model, dev_dataloader, device,
                                                                          label_list, config.task_name,
                                                                          use_crf=config.use_crf)
                    print("......" * 10)
                    print("checkpoint: " + str(int((global_step + 1) / config.checkpoint)))
                    print("Loss: Total")
                    print(loss)
                    print("......" * 10)
                    print("CWS: precision, recall, f1")
                    print(c_p, c_r, c_f1)
                    print("......" * 10)
                    print("NER: precision, recall, f1")
                    print(p_p, p_r, p_f1)

                    if loss < best_loss:
                        best_loss = loss
                        if config.export_model:
                            # export a better model
                            model_to_save = model.module if hasattr(model, "module") else model
                            output_model_file = os.path.join(config.output_dir, config.output_model_name)
                            torch.save(model_to_save.state_dict(), output_model_file)
                    if c_f1 > cws_best_f1:
                        cws_best_precision = c_p
                        cws_best_recall = c_r
                        cws_best_f1 = c_f1
                    if p_f1 > ner_best_f1:
                        pos_best_precision = p_p
                        pos_best_recall = p_r
                        ner_best_f1 = p_f1
                    # end of if do_eval

                print("-*-" * 15, flush=True)
                model.train()
                # end of checkpoint

        epoch_finish = time.time()
        print("EPOCH: %d; TIME: %.2fs" % (idx, epoch_finish - epoch_start), flush=True)
        # end of epoch

    train_finish = time.time()
    print("TOTAL_TIME: %.2fs" % (train_finish - train_start))

    print("=&=" * 15)
    print("CWS+POS: best loss ")
    print(best_loss)
    print("CWS: best p, r, f1 ")
    print(cws_best_precision, cws_best_recall, cws_best_f1)
    print("POS: best p, r, f1 ")
    print(pos_best_precision, pos_best_recall, pos_best_f1)
    print("=&=" * 15)
