import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.jit as jit
import math

from locked_dropout import LockedDropout


class LinearDropConnect(jit.ScriptModule):
    def __init__(self, in_features, out_features, bias=True, dropout=0.):
        super(LinearDropConnect, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(0))
        torch.nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def sample_mask(self):
        if self.dropout == 0.:
            self._weight = self.weight
        else:
            mask = self.weight.new_empty(
                self.weight.size(),
                dtype=torch.uint8
            )
            mask.bernoulli_(self.dropout)
            self._weight = self.weight.masked_fill(mask, 0.)

    def forward(self, input, sample_mask=False):
        if self.training:
            if sample_mask:
                self.sample_mask()
            return torch.matmul(input, self._weight) + self.bias
        else:
            return (torch.matmul(input, self.weight * (1 - self.dropout)) +
                    self.bias)


def cumsoftmax(x, dim=-1):
    return torch.cumsum(F.softmax(x, dim=dim), dim=dim)


class ONLSTMCell(jit.ScriptModule):

    def __init__(self, input_size, hidden_size, chunk_size, dropconnect=0.):
        super(ONLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.n_chunk = int(hidden_size / chunk_size)

        self.ih_w = nn.Parameter(torch.randn(
            input_size,
            4 * hidden_size + self.n_chunk * 2
        ))
        self.ih_b = nn.Parameter(torch.randn(
            4 * hidden_size + self.n_chunk * 2
        ))

        # TODO: put back dropconnect

        self.hh = LinearDropConnect(hidden_size, hidden_size*4+self.n_chunk*2,
                                    bias=True, dropout=dropconnect)
        self.drop_weight_modules = [self.hh]

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.ih_w.size(0))
        torch.nn.init.uniform_(self.ih_w, -bound, bound)
        if self.bias is not None:
            torch.nn.init.uniform_(self.ih_b, -bound, bound)

    def ih(self, input):
        return torch.matmul(input, self.ih_w) + self.ih_b

    def forward(self, input, hidden,
                transformed_input=None):
        hx, cx = hidden
        batch_size = hx.size(0)
        if transformed_input is None:
            transformed_input = self.ih(input)

        gates = transformed_input + self.hh(hx)

        cingate, cforgetgate, gates = \
            gates.split([self.n_chunk, self.n_chunk,
                         4 * self.hidden_size], dim=1)
        outgate, cell, ingate, forgetgate = \
            gates.view(batch_size,
                       self.n_chunk*4, self.chunk_size).chunk(4, 1)

        cingate = 1. - cumsoftmax(cingate)
        cforgetgate = cumsoftmax(cforgetgate)

        distance_cforget = 1. - cforgetgate.sum(dim=-1) / self.n_chunk
        distance_cin = cingate.sum(dim=-1) / self.n_chunk

        cingate = cingate[:, :, None]
        cforgetgate = cforgetgate[:, :, None]

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cell = torch.tanh(cell)
        outgate = torch.sigmoid(outgate)

        # cy = cforgetgate * forgetgate * cx + cingate * ingate * cell

        overlap = cforgetgate * cingate
        forgetgate = forgetgate * overlap + (cforgetgate - overlap)
        ingate = ingate * overlap + (cingate - overlap)
        cy = forgetgate * cx + ingate * cell
        hy = outgate * torch.tanh(cy)

        # hy = outgate * F.tanh(self.c_norm(cy))
        return hy.view(-1, self.hidden_size), cy, (distance_cforget, distance_cin)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(bsz, self.hidden_size).zero_(),
                weight.new(bsz, self.n_chunk, self.chunk_size).zero_())

    def sample_masks(self):
        for m in self.drop_weight_modules:
            m.sample_mask()


class ONLSTMStack(nn.Module):
    def __init__(self, layer_sizes, chunk_size, dropout=0., dropconnect=0.):
        super(ONLSTMStack, self).__init__()
        self.cells = nn.ModuleList([ONLSTMCell(layer_sizes[i],
                                               layer_sizes[i+1],
                                               chunk_size,
                                               dropconnect=dropconnect)
                                    for i in range(len(layer_sizes) - 1)])
        self.lockdrop = LockedDropout()
        self.dropout = dropout
        self.sizes = layer_sizes

    def init_hidden(self, bsz):
        return [c.init_hidden(bsz) for c in self.cells]

    def forward(self, input, hidden):
        length, batch_size, _ = input.size()

        if self.training:
            for c in self.cells:
                c.sample_masks()

        prev_state = list(hidden)
        prev_layer = input

        raw_outputs = [None] * len(self.cells)
        outputs = [None] * len(self.cells)
        distances_forget = [None] * len(self.cells)
        distances_in = [None] * len(self.cells)
        for l in range(len(self.cells)):
            curr_layer = [None] * length
            dist = [None] * length
            t_input = self.cells[l].ih(prev_layer)

            for t in range(length):
                hidden, cell, d = self.cells[l](
                    None, prev_state[l],
                    transformed_input=t_input[t]
                )
                prev_state[l] = hidden, cell  # overwritten every timestep
                curr_layer[t] = hidden
                dist[t] = d

            prev_layer = torch.stack(curr_layer)
            dist_cforget, dist_cin = zip(*dist)
            dist_layer_cforget = torch.stack(dist_cforget)
            dist_layer_cin = torch.stack(dist_cin)

            raw_outputs[l] = prev_layer
            if l < len(self.cells) - 1:
                prev_layer = self.lockdrop(prev_layer, self.dropout)
            outputs[l] = prev_layer
            distances_forget[l] = dist_layer_cforget
            distances_in[l] = dist_layer_cin
        output = prev_layer

        return output, prev_state, raw_outputs, outputs, (torch.stack(distances_forget), torch.stack(distances_in))


if __name__ == "__main__":
    x = torch.Tensor(10, 10, 10)
    x.data.normal_()
    lstm = LSTMCellStack([10, 10, 10])
    print(lstm(x, lstm.init_hidden(10))[1])

