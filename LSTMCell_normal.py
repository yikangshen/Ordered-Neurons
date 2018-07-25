import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import RNNCellBase


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class LSTMCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, dropout=0):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.ih = nn.Sequential(nn.Linear(input_size, 4 * hidden_size, bias=True), LayerNorm(4 * hidden_size))
        self.hh = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size, bias=True), LayerNorm(4 * hidden_size))

        self.c_norm = LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, input, hidden, rmask):
        hx, cx = hidden

        input = self.drop(input)
        hx = hx * rmask
        gates = self.ih(input) + self.hh(hx) #+ self.bias

        cell, ingate, forgetgate, outgate = gates.chunk(4, 1)

        distance = forgetgate.sum(dim=-1) / self.hidden_size

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cell = F.tanh(cell)
        outgate = F.sigmoid(outgate)

        cy = forgetgate * cx + ingate * cell
        hy = outgate * F.tanh(self.c_norm(cy))

        return hy, cy, distance

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return weight.new(bsz, self.hidden_size).zero_(), \
               weight.new(bsz, self.hidden_size).zero_()