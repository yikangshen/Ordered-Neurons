import argparse

import nltk
import numpy
import torch
from torch.autograd import Variable

import data_ptb
import data


# Test model
def build_tree(depth, sen):
    assert len(depth) == len(sen)

    if len(depth) == 1:
        parse_tree = sen[0]
    else:
        idx_max = numpy.argmax(depth)
        parse_tree = []
        if len(sen[:idx_max]) > 0:
            tree0 = build_tree(depth[:idx_max], sen[:idx_max])
            parse_tree.append(tree0)
        tree1 = sen[idx_max]
        if len(sen[idx_max + 1:]) > 0:
            tree2 = build_tree(depth[idx_max + 1:], sen[idx_max + 1:])
            tree1 = [tree1, tree2]
        if parse_tree == []:
            parse_tree = tree1
        else:
            parse_tree.append(tree1)
    return parse_tree


# def build_tree(depth, sen):
#     assert len(depth) == len(sen)
#     assert len(depth) >= 0
#     depth[0] = 0
#
#     if len(depth) == 1:
#         parse_tree = sen[0]
#     else:
#         idx_max = numpy.argmax(depth)
#         parse_tree = []
#         if len(sen[:idx_max]) > 0:
#             tree0 = build_tree(depth[:idx_max], sen[:idx_max])
#             parse_tree.append(tree0)
#         if len(sen[idx_max:]) > 0:
#             tree1 = build_tree(depth[idx_max:], sen[idx_max:])
#             parse_tree.append(tree1)
#     return parse_tree


def get_brackets(tree, idx=0):
    brackets = set()
    if isinstance(tree, list) or isinstance(tree, nltk.Tree):
        for node in tree:
            node_brac, next_idx = get_brackets(node, idx)
            if next_idx - idx > 1:
                brackets.add((idx, next_idx))
                brackets.update(node_brac)
            idx = next_idx
        return brackets, idx
    else:
        return brackets, idx + 1


def mean(x):
    return sum(x) / len(x)


def test(model, corpus, cuda, prt=False):
    model.eval()

    prec_list = []
    reca_list = []
    f1_list = []

    nsens = 0
    for sen, sen_tree in zip(corpus.train_sens, corpus.train_trees):
        if args.wsj10 and len(sen) > 12:
            continue
        x = numpy.array([corpus.dictionary[w] for w in sen])
        input = Variable(torch.LongTensor(x[:, None]))
        if cuda:
            input = input.cuda()

        hidden = model.init_hidden(1)
        _, hidden = model(input, hidden)

        gates = model.distance.squeeze().data.cpu().numpy().mean(axis=1)

        depth = gates[1:-1]
        sen = sen[1:-1]
        parse_tree = build_tree(depth, sen)

        model_out, _ = get_brackets(parse_tree)
        std_out, _ = get_brackets(sen_tree)
        overlap = model_out.intersection(std_out)

        prec = float(len(overlap)) / (len(model_out) + 1e-8)
        reca = float(len(overlap)) / (len(std_out) + 1e-8)
        if len(std_out) == 0:
            reca = 1.
            if len(model_out) == 0:
                prec = 1.
        f1 = 2 * prec * reca / (prec + reca + 1e-8)
        prec_list.append(prec)
        reca_list.append(reca)
        f1_list.append(f1)

        nsens += 1
        if prt and nsens % 100 == 0:
            # for i in range(len(sen)):
            #     print '%15s\t%.2f\t%s' % (sen[i], depth[i], str(attentions[i, 1]))
            print('Model output:')
            print(parse_tree)
            print(model_out)
            print('Standard output:')
            print(sen_tree)
            print(std_out)
            print('Prec: %f, Reca: %f, F1: %f' % (prec, reca, f1))
            print('-' * 80)

    if prt:
        print('-' * 80)
        print('Mean Prec: %f, Mean Reca: %f, Mean F1: %f' % (mean(prec_list), mean(reca_list), mean(f1_list)))
        print('Number of sentence: %i' % nsens)

    return mean(f1_list)


if __name__ == '__main__':
    marks = [' ', '-', '=']

    numpy.set_printoptions(precision=2, suppress=True, linewidth=5000)

    parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

    # Model parameters.
    parser.add_argument('--data', type=str, default='./data/ptb',
                        help='location of the data corpus')
    parser.add_argument('--checkpoint', type=str, default='./model/PTB.pt',
                        help='model checkpoint to use')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--wsj10', action='store_true',
                        help='use WSJ10')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    # Load model
    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f)
        model.cpu()
        if args.cuda:
            model.cuda()
            torch.cuda.manual_seed(args.seed)

    # Load data
    corpus = data_ptb.Corpus(args.data)
    corpus.dictionary = data.Corpus('./data/penn').dictionary

    test(model, corpus, args.cuda, prt=True)