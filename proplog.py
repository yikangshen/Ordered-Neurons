import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

import ON_LSTM

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/propositionallogic/',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=128,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=400,
                    help='number of hidden units per layer')
parser.add_argument('--chunk_size', type=int, default=10,
                    help='number of units per chunk')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=1.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--std', action='store_true',
                    help='use standard LSTM')
# parser.add_argument('--resume', type=str,  default='',
#                     help='path of model to resume')
args = parser.parse_args()
args.tied = True

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################


def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, optimizer], f)


def model_load(fn):
    global model, optimizer
    with open(fn, 'rb') as f:
        model, optimizer = torch.load(f)


class LogicInference(object):
    def __init__(self, datapath='data/propositionallogic/', maxn=12):
        """maxn=0 indicates variable expression length."""
        self.num2char = ['(', ')',
                         'a', 'b', 'c', 'd', 'e', 'f',
                         'or', 'and', 'not']
        self.char2num = {self.num2char[i]: i
                         for i in range(len(self.num2char))}

        self.num2lbl = list('<>=^|#v')
        self.lbl2num = {self.num2lbl[i]: i
                        for i in range(len(self.num2lbl))}

        self.train_set, self.valid_set, self.test_set = [], [], []
        counter = 0
        for i in range(maxn):
            itrainexample = self._readfile(os.path.join(datapath, "train" + str(i)))
            for e in itrainexample:
                counter += 1
                if counter % 10 == 0:
                    self.valid_set.append(e)
                else:
                    self.train_set.append(e)
                # self.train_set = self.train_set + itrainexample

        for i in range(13):
            itestexample = self._readfile(os.path.join(datapath, "test" + str(i)))
            self.test_set.append(itestexample)

    def _readfile(self, filepath):
        f = open(filepath, 'r')
        examples = []
        for line in f.readlines():
            relation, p1, p2 = line.strip().split('\t')
            p1 = p1.split()
            p2 = p2.split()
            examples.append((self.lbl2num[relation],
                             [self.char2num[w] for w in p1],
                             [self.char2num[w] for w in p2]))
        return examples

    def stream(self, dataset, batch_size, shuffle=False, pad=None):
        if pad is None:
            pad = len(self.num2char)
        import random
        import math
        batch_count = int(math.ceil(len(dataset) / float(batch_size)))

        def shuffle_stream():
            if shuffle:
                random.shuffle(dataset)
            for i in range(batch_count):
                yield dataset[i * batch_size: (i+1) * batch_size]

        def arrayify(stream, pad):
            for batch in stream:
                batch_lbls = np.array([x[0] for x in batch], dtype=np.int64)
                batch_sent = [x[1] for x in batch] + [x[2] for x in batch]
                max_len = max(len(s) for s in batch_sent)
                batch_idxs = np.full((max_len, len(batch_sent)), pad,
                                     dtype=np.int64)
                for i in range(len(batch_sent)):
                    sentence = batch_sent[i]
                    batch_idxs[:len(sentence), i] = sentence
                yield batch_idxs, batch_lbls

        stream = shuffle_stream()
        stream = arrayify(stream, pad)
        return stream

corpus = LogicInference(maxn=7)

###############################################################################
# Build the model
###############################################################################
###
# if args.resume:
#    print('Resuming model ...')
#    model_load(args.resume)
#    optimizer.param_groups[0]['lr'] = args.lr
#    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
#    if args.wdrop:
#        for rnn in model.rnn.cells:
#            rnn.hh.dropout = args.wdrop
###

ntokens = len(corpus.num2char) + 1
nlbls = len(corpus.num2lbl)


class Classifier(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, nout, chunk_size, dropout, wdrop):
        super(Classifier, self).__init__()

        self.padding_idx = ntoken - 1
        self.embedding = nn.Embedding(ntoken, ninp,
                                      padding_idx=self.padding_idx)
        self.encoder = ON_LSTM.ONLSTMStack(
            layer_sizes=[ninp] + [nhid] * nlayers,
            chunk_size=chunk_size,
            dropout=dropout,
            dropconnect=wdrop,
            std=args.std
        )

        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4 * nhid, nhid),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, nout),
        )

        self.drop = nn.Dropout(dropout)

        self.cost = nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        batch_size = input.size(1)
        lengths = (input != self.padding_idx).sum(0)
        emb = self.drop(self.embedding(input))
        hidden = self.encoder.init_hidden(batch_size)
        output, _, _, _, _ = self.encoder(emb, hidden)
        last_hidden = output[lengths - 1, torch.arange(batch_size).long()]
        clause_1 = last_hidden[:batch_size // 2]
        clause_2 = last_hidden[batch_size // 2:]
        output = self.mlp(torch.cat([clause_1, clause_2,
                                     clause_1 * clause_2,
                                     torch.abs(clause_1 - clause_2)], dim=1))
        return output

model = Classifier(
    ntoken=ntokens,
    ninp=args.emsize,
    nhid=args.nhid,
    nlayers=args.nlayers,
    nout=nlbls,
    chunk_size=args.chunk_size,
    dropout=args.dropout,
    wdrop=args.wdrop
)


if args.cuda:
    model = model.cuda()
###
params = list(model.parameters())
total_params = sum(x.size()[0] * x.size()[1]
                   if len(x.size()) > 1 else x.size()[0]
                   for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

###############################################################################
# Training code
###############################################################################


def valid():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    total_datapoints = 0
    for sents, lbls in corpus.stream(corpus.valid_set, args.batch_size):
        count = lbls.shape[0]
        sents = torch.from_numpy(sents)
        lbls = torch.from_numpy(lbls)
        if args.cuda:
            sents = sents.cuda()
            lbls = lbls.cuda()
        lin_output = model(sents)
        total_loss += torch.sum(
            torch.argmax(lin_output, dim=1) == lbls
        ).float().data
        total_datapoints += count
        accs = total_loss.item() / total_datapoints
    return accs


def evaluate():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    accs = []
    for l in range(13):
        total_loss = 0
        total_datapoints = 0
        for sents, lbls in corpus.stream(corpus.test_set[l], args.batch_size):
            count = lbls.shape[0]
            sents = torch.from_numpy(sents)
            lbls = torch.from_numpy(lbls)
            if args.cuda:
                sents = sents.cuda()
                lbls = lbls.cuda()
            lin_output = model(sents)
            total_loss += torch.sum(
                torch.argmax(lin_output, dim=1) == lbls
            ).float().data
            total_datapoints += count
        accs.append(total_loss.item() / total_datapoints)
    return accs


def train():
    # Turn on training mode which enables dropout.
    total_loss = 0
    total_acc = 0
    start_time = time.time()
    batch = 0
    for sents, lbls in corpus.stream(corpus.train_set, args.batch_size,
                                     shuffle=True):
        sents = torch.from_numpy(sents)
        lbls = torch.from_numpy(lbls)
        if args.cuda:
            sents = sents.cuda()
            lbls = lbls.cuda()

        model.train()
        optimizer.zero_grad()

        lin_output = model(sents)
        loss = model.cost(lin_output, lbls)
        acc = torch.mean(
            (torch.argmax(lin_output, dim=1) == lbls).float())
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip:
            torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += loss.detach().data
        total_acc += acc.detach().data
        if batch % args.log_interval == 0 and batch > 0:
            elapsed = time.time() - start_time
            print(
                '| epoch {:3d} '
                '| lr {:05.5f} | ms/batch {:5.2f} '
                '| loss {:5.2f} | acc {:0.2f}'.format(
                    epoch,
                    optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval,
                    total_loss.item() / args.log_interval,
                    total_acc.item() / args.log_interval))
            total_loss = 0
            total_acc = 0
            start_time = time.time()
        ###
        batch += 1

# Loop over epochs.
lr = args.lr
stored_loss = 0.

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    optimizer = torch.optim.Adam(params,
                                 lr=args.lr,
                                 # betas=(0, 0.999),
                                 eps=1e-9,
                                 weight_decay=args.wdecay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.5, patience=2, threshold=0)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = valid()
        test_loss = evaluate()

        print('-' * 89)
        print(
            '| end of epoch {:3d} '
            '| time: {:5.2f}s '
            '| valid acc: {:.2f} '
            '|\n'.format(
                epoch,
                (time.time() - epoch_start_time),
                val_loss
            ),
            ', '.join(str('{:0.2f}'.format(v)) for v in test_loss)
        )

        if val_loss > stored_loss:
            model_save(args.save)
            print('Saving model (new best validation)')
            stored_loss = val_loss
        print('-' * 89)

        scheduler.step(val_loss)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save)
test_loss = evaluate()
val_loss = valid()
print('-' * 89)
print(
    '| valid acc: {:.2f} '
    '|\n'.format(
        val_loss
    ),
    ', '.join(str('{:0.2f}'.format(v)) for v in test_loss)
)
