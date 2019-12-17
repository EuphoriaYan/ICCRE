"""Implementation of batch-normalized LSTM."""

import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-5])
if root_path not in sys.path:
    sys.path.insert(0, root_path)  

device = "cuda:3"

import torch
import torch.autograd as autograd
from torch import nn
from torch.nn import init


import logging
import numpy as np 

logger = logging.getLogger(__name__)


class WordLSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(WordLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 3 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        init.orthogonal_(self.weight_ih)
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh = nn.Parameter(weight_hh_data)
        # The bias is just set to zero vectors.
        if self.use_bias:
            init.constant_(self.bias, val=0)

    def forward(self, input_, hx):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        f, i, g = torch.split(wh_b + wi, self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f)*c_0 + torch.sigmoid(i)*torch.tanh(g)
        return c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class MultiInputLSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """
        super(MultiInputLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 3 * hidden_size))
        self.alpha_weight_ih = nn.Parameter(torch.FloatTensor(input_size, hidden_size))
        self.alpha_weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
            self.alpha_bias = nn.Parameter(torch.FloatTensor(hidden_size))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('alpha_bias', None)
        self.init_parameters()

    def init_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        init.orthogonal_(self.weight_ih)
        init.orthogonal_(self.alpha_weight_ih)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh = nn.Parameter(weight_hh_data)

        alpha_weight_hh_data = torch.eye(self.hidden_size)
        alpha_weight_hh_data = alpha_weight_hh_data.repeat(1, 1)
        self.alpha_weight_hh = nn.Parameter(alpha_weight_hh_data)

        # The bias is just set to zero vectors.
        if self.use_bias:
            init.constant_(self.bias, val=0)
            init.constant_(self.alpha_bias, val=0)

    def forward(self, input_, c_input, hx):
        """
        Args:
            batch = 1
            input_: A (batch, input_size) tensor containing input
                features.
            c_input: A  list with size c_num,each element is the input ct from skip word (batch, hidden_size).
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        h_0 = h_0.to(device)
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        i, o, g = torch.split(wh_b + wi, self.hidden_size, dim=1)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        # c_input (batch_size, hidden_dim)
        # c_input_var = torch.cat(c_input, 0)
        c_input_var = c_input.squeeze(1)  ## (c_num(1), batch_size, hidden_dim)
        # alpha_wi = torch.addmm(self.alpha_bias, input_, self.alpha_weight_ih).expand(1, self.hidden_size)
        alpha_wi = torch.addmm(self.alpha_bias, input_, self.alpha_weight_ih)
        alpha_wh = torch.mm(c_input_var, self.alpha_weight_hh)
        alpha = torch.sigmoid(alpha_wi + alpha_wh)
        i = i.unsqueeze(1)
        alpha = alpha.unsqueeze(1)
        alpha = torch.exp(torch.cat([i, alpha], 1))
        alpha_sum = alpha.sum(1)

        for idx in range(batch_size):
            for jdx in range(self.hidden_size):
                alpha[idx][0][jdx] = torch.div(alpha[idx][0][jdx], alpha_sum[idx][jdx])
                alpha[idx][1][jdx] = torch.div(alpha[idx][1][jdx], alpha_sum[idx][jdx])

        g = g.unsqueeze(1)
        c_input_var = c_input_var.unsqueeze(1)
        merge_i_c = torch.cat([g, c_input_var], 1)
        c_1 = merge_i_c * alpha
        c_1 = c_1.sum(1)
        h_1 = o * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class LatticeLSTM(nn.Module):

    """A module that runs multiple steps of LSTM."""

    def __init__(self, input_dim, hidden_dim, word_drop, word_alphabet_size, word_emb_dim,
                 pretrain_word_emb=None,
                 left2right=True,
                 fix_word_emb=True,
                 use_bias=True):
        super(LatticeLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_emb = nn.Embedding(word_alphabet_size, word_emb_dim)
        if pretrain_word_emb is not None:
            self.word_emb.weight = nn.Parameter(torch.FloatTensor(pretrain_word_emb))
        else:
            self.word_emb.weight = nn.Parameter(torch.FloatTensor(self.random_embedding(word_alphabet_size, word_emb_dim)))
        if fix_word_emb:
            self.word_emb.weight.requires_grad = False
        
        self.word_dropout = nn.Dropout(word_drop)

        self.rnn = MultiInputLSTMCell(input_dim, hidden_dim)
        self.word_rnn = WordLSTMCell(word_emb_dim, hidden_dim)
        self.left2right = left2right

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, input, skip_input, hidden=None):
        """
            input: variable (batch, seq_len), batch = 1
            skip_input_list: [skip_input, volatile_flag]
            skip_input: three dimension list, with length is seq_len. Each element is a list of matched word id and its length. 
                        example: [[], [[25,13],[2,3]]] 25/13 is word id, 2,3 is word length . 
        """

        if not self.left2right:
            skip_input = convert_forward_gaz_to_backward(skip_input)

        input = input.transpose(1, 0)
        skip_input = skip_input.permute(1, 2, 0)
        # skip_input (seq_len, 2, batch_size)

        seq_len = input.size(0)
        batch_size = input.size(1)

        hidden_out = []
        memory_out = []
        if hidden:
            (hx, cx) = hidden
        else:
            hx = nn.Parameter(torch.zeros(batch_size, self.hidden_dim))
            cx = nn.Parameter(torch.zeros(batch_size, self.hidden_dim))

        id_list = list(range(seq_len))
        if not self.left2right:
            id_list = list(reversed(id_list))

        # input_c_list = [[list() for i in range(batch_size)] for j in range(seq_len)]
        input_c_list = torch.FloatTensor(np.zeros((seq_len, batch_size, self.hidden_dim))).to(device)
        # input_c_list (seq_len, batch_size, hidden_dim)
        for t in id_list:
            (hx, cx) = self.rnn(input[t], input_c_list[t], (hx, cx))
            hidden_out.append(hx)
            memory_out.append(cx)
            # skip_input (seq_len, 2, batch_size)
            word_var = skip_input[t][0]
            word_emb = self.word_emb(word_var)
            word_emb = self.word_dropout(word_emb)
            ct = self.word_rnn(word_emb, (hx, cx))
            # ct (batch_size, hidden_size)
            assert(ct.size(0) == len(skip_input[t][1]))

            length = skip_input[t][1]
            for i in range(batch_size):
                skip_feature = ct[i]
                if self.left2right:
                    input_c_list[t+length-1][i] = skip_feature
                else:
                    input_c_list[t-length+1][i] = skip_feature

        if not self.left2right:
            hidden_out = list(reversed(hidden_out))
            memory_out = list(reversed(memory_out))

        output_hidden, output_memory = torch.cat(hidden_out, 0), torch.cat(memory_out, 0)
        return output_hidden.unsqueeze(0), output_memory.unsqueeze(0)


def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0, size):
        list_of_objects.append(list())
    return list_of_objects


def convert_forward_gaz_to_backward(forward_gaz):
    length = len(forward_gaz)
    backward_gaz = init_list_of_objects(length)
    for idx in range(length):
        if forward_gaz[idx]:
            assert(len(forward_gaz[idx]) == 2)
            num = len(forward_gaz[idx][0])
            for idy in range(num):
                the_id = forward_gaz[idx][0][idy]
                the_length = forward_gaz[idx][1][idy]
                new_pos = idx + the_length - 1
                if backward_gaz[new_pos]:
                    backward_gaz[new_pos][0].append(the_id)
                    backward_gaz[new_pos][1].append(the_length)
                else:
                    backward_gaz[new_pos] = [[the_id],[the_length]]
    return backward_gaz



