import torch
import torch.nn as nn

from LSTMCell_new import LSTMCell


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers,
                 nslots=5, nlookback=1,
                 dropout=0.4, idropout=0.4, rdropout=0.1,
                 tie_weights=False):
        super(RNNModel, self).__init__()

        self.nhid = nhid
        self.ninp = ninp
        self.nlayers = nlayers
        self.nslots = nslots
        self.nlookback = nlookback

        self.drop = nn.Dropout(dropout)
        self.idrop = nn.Dropout(idropout)
        self.rdrop = nn.Dropout(rdropout)

        # Feedforward layers
        self.encoder = nn.Embedding(ntoken, ninp)
        self.reader = nn.ModuleList([LSTMCell(ninp if i == 0 else nhid,
                                              ninp if i == (nlayers - 1) else nhid,
                                              dropout=dropout if i == 0 else idropout)
                                     for i in range(0, nlayers)]
                                    )
        self.decoder = nn.Linear(ninp, ntoken)

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.distance = None

        self.init_weights()

    def init_weights(self):
        initrange = 0.01
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden_states):
        ntimestep = input.size(0)
        bsz = input.size(1)
        emb = self.encoder(input)  # timesteps, bsz, ninp
        output_h = []

        # distance, parser_state = self.parser(emb, parser_state)
        distance = []

        rmask = []
        for i in range(self.nlayers):
            rmask_i = torch.autograd.Variable(torch.ones(self.ninp if i == (self.nlayers - 1) else self.nhid))
            if input.is_cuda: rmask_i = rmask_i.cuda()
            rmask_i = self.rdrop(rmask_i)
            rmask.append(rmask_i)

        for i in range(input.size(0)):
            emb_i = emb[i]  # emb_i: bsz, nhid
            h_list = []

            # summarize layer
            h_i = emb_i
            distance_i = []
            for j in range(self.nlayers):
                hidden = hidden_states[j]

                h_i, c_i, distance_ij = self.reader[j](h_i, hidden, rmask[j])
                distance_i.append(distance_ij)

                # updata states
                h_list.append(h_i)
                hidden_states[j] = (h_i, c_i)

            if not self.training:
                distance_i = torch.stack(distance_i, dim=-1)
                distance.append(distance_i)
            output_h.append(h_i)

        if not self.training:
            self.distance = torch.stack(distance, dim=1)

        output = torch.stack(output_h, dim=0)

        # output = self.predictor(output.view(ntimestep * bsz, -1))

        output = self.drop(output)
        decoded = self.decoder(output.view(ntimestep * bsz, -1))
        return decoded.view(ntimestep, bsz, -1), hidden_states

    def init_hidden(self, bsz):
        return [self.reader[i].init_hidden(bsz)
                for i in range(self.nlayers)]
