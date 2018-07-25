import argparse
import numpy
import torch

import data

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = numpy.exp(x - numpy.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

marks = [' ', '-', '=']

numpy.set_printoptions(precision=2, suppress=True, linewidth=5000)

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model/model_LM.pt',
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

if args.checkpoint is None:
    checkpoint = input('Input checkpoint name:')
    args.checkpoint = './model/' + checkpoint


# def build_tree(depth, sen):
#     assert len(depth) == len(sen)
#
#     if len(depth) == 1:
#         parse_tree = sen[0]
#     else:
#         idx_max = numpy.argmax(depth)
#         parse_tree = []
#         if len(sen[:idx_max]) > 0:
#             tree0 = build_tree(depth[:idx_max], sen[:idx_max])
#             parse_tree.append(tree0)
#         tree1 = sen[idx_max]
#         if len(sen[idx_max + 1:]) > 0:
#             tree2 = build_tree(depth[idx_max + 1:], sen[idx_max + 1:])
#             tree1 = [tree1, tree2]
#         if parse_tree == []:
#             parse_tree = tree1
#         else:
#             parse_tree.append(tree1)
#     return parse_tree

def build_tree(depth, sen):
    assert len(depth) == len(sen)
    assert len(depth) >= 0
    depth[0] = 0

    if len(depth) == 1:
        parse_tree = sen[0]
    else:
        idx_max = numpy.argmax(depth)
        parse_tree = []
        if len(sen[:idx_max]) > 0:
            tree0 = build_tree(depth[:idx_max], sen[:idx_max])
            parse_tree.append(tree0)
        if len(sen[idx_max:]) > 0:
            tree1 = build_tree(depth[idx_max:], sen[idx_max:])
            parse_tree.append(tree1)
    return parse_tree

def MRG(tr):
    if isinstance(tr, str):
        return '(' + tr + ')'
        # return tr + ' '
    else:
        s = '('
        for subtr in tr:
            s += MRG(subtr)
        s += ')'
        return s

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        pass
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()
print(model)

if args.cuda:
    model.cuda()
else:
    model.cpu()
marks = marks[:model.nlayers]
marks = marks[::-1]

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)

while True:
    sens = input('Input a sentences:')
    hidden = model.init_hidden(1)
    for s in sens.split('\t'):
        words = s.strip().split()
        x = numpy.array([corpus.dictionary[w] for w in words])
        Vx = torch.LongTensor(x[:, None])
        
        # hidden = model.init_hidden(1)
        output, hidden = model(Vx, hidden)
        output = output.squeeze().data.numpy()[:-1]
        output = numpy.log(softmax(output))
        output = numpy.pad(output, ((1, 0), (0, 0)), 'constant', constant_values=0)
        output = numpy.exp(-output[range(len(words)), x])

        if not model.distance is None:
            distance = model.distance.squeeze(0).data.numpy()
        else:
            distance = numpy.zeros(len(words))
        for i in range(len(words)):
            print('%15s\t%10.1f\t%s' % (words[i], output[i], str(distance[i])))

        print(output[1:].mean())

        for i in range(distance.shape[1]):
            parse_tree = build_tree(distance[:, i].copy(), words)
            print(MRG(parse_tree))
        parse_tree = build_tree(distance.mean(axis=1), words)
        print(MRG(parse_tree))
