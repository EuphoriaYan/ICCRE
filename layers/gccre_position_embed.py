#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import json
import math
import shutil
import tarfile
import logging
import tempfile
import random
import argparse

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np

from layers.bert_basic_model import *
from utils.tokenization import BertTokenizer
from layers.gccre_embedder import GccreEmbedding


WEIGHTS_NAME = "gccre_pos_emb_model.bin"


class GccrePositionEmbedder(nn.Module):
    def __init__(self, config):
        super(GccrePositionEmbedder, self).__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.output_size)

        token_tool = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=False)
        idx2tokens = token_tool.ids_to_tokens
        self.gccre_encoder = GccreEmbedding(config, idx2tokens)
        self.layer_norm = BertLayerNorm(config.output_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        gccre_embeddings, gccre_cls_loss = self.gccre_encoder(input_ids)

        embeddings = position_embeddings + gccre_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, gccre_cls_loss

    @classmethod
    def from_pretrained(cls, gccre_config):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name:
                - a path or url to a pretrained model archive containing:
                    . `gccre_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a GccreForPretrained instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """

        archive_file = gccre_config.gccre_pos_emb_model
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file)
        except FileNotFoundError:
            logger.error("We assumed '{}' was a path, "
                         "but couldn't find any file associated to this path."
                         .format(archive_file)
            )
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(archive, tempdir)
            serialization_dir = tempdir

        # Load config
        logger.info("Model config {}".format(gccre_config))
        # Instantiate model.
        model = cls(gccre_config)
        # model = cls(input_configs, *inputs, **kwargs)
        weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)

        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict,
                                         prefix,
                                         local_metadata,
                                         True,
                                         missing_keys,
                                         unexpected_keys,
                                         error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='')
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        return model


if __name__ == "__main__":
    pass