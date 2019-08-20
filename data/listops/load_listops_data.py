from collections import namedtuple
from spinn import util

from spinn.data.listops.base import FIXED_VOCABULARY

SENTENCE_PAIR_DATA = False
OUTPUTS = list(range(10))
LABEL_MAP = {str(x): i for i, x in enumerate(OUTPUTS)}

Node = namedtuple('Node', 'tag span')


def spans(transitions, tokens=None):
    n = (len(transitions) + 1) // 2
    stack = []
    buf = [Node("leaf", (l, r)) for l, r in zip(list(range(n)), list(range(1, n + 1)))]
    buf = list(reversed(buf))

    nodes = []
    reduced = [False] * n

    def SHIFT(item):
        nodes.append(item)
        return item

    def REDUCE(l, r):
        tag = None
        i = r.span[1] - 1
        if tokens is not None and tokens[i] == ']' and not reduced[i]:
            reduced[i] = True
            tag = "struct"
        new_stack_item = Node(tag=tag, span=(l.span[0], r.span[1]))
        nodes.append(new_stack_item)
        return new_stack_item

    for t in transitions:
        if t == 0:
            stack.append(SHIFT(buf.pop()))
        elif t == 1:
            r, l = stack.pop(), stack.pop()
            stack.append(REDUCE(l, r))

    return nodes


def load_data(path, lowercase=None, choose=lambda x: True, eval_mode=False):
    examples = []
    with open(path) as f:
        for example_id, line in enumerate(f):
            line = line.strip()
            label, seq = line.split('\t')
            if len(seq) <= 1:
                continue

            tokens, transitions = util.ConvertBinaryBracketedSeq(
                seq.split(' '))

            example = {}
            example["label"] = label
            example["sentence"] = seq
            example["tokens"] = tokens
            example["transitions"] = transitions
            example["example_id"] = str(example_id)

            examples.append(example)
    return examples
