###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import numpy

import torch
from torch.autograd import Variable

import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='model/summ_pred_s10_l1_b2.pt',
                    help='model checkpoint to use')
parser.add_argument('--words', type=int, default='30',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.data)

while True:
    hidden = model.init_hidden(1)
    sen = raw_input('Input a sentences:')
    words = sen.strip().split()
    x = numpy.array([corpus.dictionary[w] for w in words])
    input = Variable(torch.LongTensor(x[:, None]))
    _, hidden = model(input, hidden)

    input = Variable(torch.zeros(1, 1).long(), volatile=True)
    input[0, 0] = corpus.dictionary['</s>']
    if args.cuda:
        input.data = input.data.cuda()

    words = []
    for i in range(args.words):
        output, hidden = model(input, hidden)
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()

        # word_idx = torch.multinomial(word_weights, 1)[0]

        word_weights[corpus.dictionary['<unk>']] = -100.
        _, word_idx = torch.max(word_weights, dim=0)
        word_idx = word_idx[0]

        input.data.fill_(word_idx)
        word = corpus.dictionary.idx2word[word_idx]

        words.append(word)

    print ' '.join(words)
    print '-' * 30
