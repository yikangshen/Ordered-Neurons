from collections import namedtuple
import random
import itertools
import time
import sys

import numpy as np

from utils import ConvertBinaryBracketedSeq
NUMBERS = list(range(10))
PADDING_TOKEN = "_PAD"
UNK_TOKEN = "_"
SENTENCE_PADDING_SYMBOL = 0

FIXED_VOCABULARY = {str(x): i + 1 for i, x in enumerate(NUMBERS)}
FIXED_VOCABULARY.update({
    PADDING_TOKEN: 0,
    "[MIN": len(FIXED_VOCABULARY) + 1,
    "[MAX": len(FIXED_VOCABULARY) + 2,
    "[FIRST": len(FIXED_VOCABULARY) + 3,
    "[LAST": len(FIXED_VOCABULARY) + 4,
    "[MED": len(FIXED_VOCABULARY) + 5,
    "[SM": len(FIXED_VOCABULARY) + 6,
    "[PM": len(FIXED_VOCABULARY) + 7,
	"[FLSUM": len(FIXED_VOCABULARY) + 8,
    "]": len(FIXED_VOCABULARY) + 9
})
assert len(set(FIXED_VOCABULARY.values())) == len(list(FIXED_VOCABULARY.values()))


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

def PreprocessDataset(
        dataset,
        vocabulary,
        seq_length,
        eval_mode=False,
        sentence_pair_data=False,
        simple=True,
        allow_cropping=False,
        pad_from_left=True):
    dataset = TrimDataset(
        dataset,
        seq_length,
        eval_mode=eval_mode,
        sentence_pair_data=sentence_pair_data,
        logger=None,
        allow_cropping=allow_cropping)
    dataset = TokensToIDs(
        vocabulary,
        dataset,
        sentence_pair_data=sentence_pair_data)

    dataset = CropAndPadSimple(
        dataset,
        seq_length,
        logger=None,
        sentence_pair_data=sentence_pair_data,
        allow_cropping=allow_cropping,
        pad_from_left=pad_from_left)

    if sentence_pair_data:
        X = np.transpose(np.array([[example["premise_tokens"] for example in dataset],
                                   [example["hypothesis_tokens"] for example in dataset]],
                                  dtype=np.int32), (1, 2, 0))
        if simple:
            transitions = np.zeros((len(dataset), 2, 0))
            num_transitions = np.transpose(np.array(
                [[len(np.array(example["premise_tokens"]).nonzero()[0]) for example in dataset],
                 [len(np.array(example["hypothesis_tokens"]).nonzero()[0]) for example in dataset]],
                dtype=np.int32), (1, 0))
        else:
            transitions = np.transpose(np.array([[example["premise_transitions"] for example in dataset],
                                                 [example["hypothesis_transitions"] for example in dataset]],
                                                dtype=np.int32), (1, 2, 0))
            num_transitions = np.transpose(np.array(
                [[example["num_premise_transitions"] for example in dataset],
                 [example["num_hypothesis_transitions"] for example in dataset]],
                dtype=np.int32), (1, 0))
    else:
        X = np.array([example["tokens"] for example in dataset],
                     dtype=np.int32)
        if simple:
            transitions = np.zeros((len(dataset), 0))
            num_transitions = np.array(
                [len(np.array(example["tokens"]).nonzero()[0]) for example in dataset],
                dtype=np.int32)
        else:
            transitions = np.array([example["transitions"]
                                    for example in dataset], dtype=np.int32)
            num_transitions = np.array(
                [example["num_transitions"] for example in dataset],
                dtype=np.int32)

    y = np.array(
        [LABEL_MAP[example["label"]] for example in dataset],
        dtype=np.int32)

    # NP Array of Strings
    example_ids = np.array([example["example_id"] for example in dataset])

    return X, transitions, y, num_transitions, example_ids


def load_data(path, lowercase=None, choose=lambda x: True, eval_mode=False):
    examples = []
    with open(path) as f:
        for example_id, line in enumerate(f):
            line = line.strip()
            label, seq = line.split('\t')
            if len(seq) <= 1:
                continue

            tokens, transitions = ConvertBinaryBracketedSeq(
                seq.split(' '))

            example = {}
            example["label"] = label
            example["sentence"] = seq
            example["tokens"] = tokens
            example["transitions"] = transitions
            example["example_id"] = str(example_id)

            examples.append(example)
    return examples


def load_data_and_embeddings(args, training_data_path, eval_data_path):
    raw_training_data = load_data(training_data_path, None, eval_mode=False)
    raw_eval_data = load_data(eval_data_path, None, eval_mode=True)
    import copy
    raw_eval_data_copy = copy.deepcopy(raw_eval_data)
    # Prepare the vocabulary
    vocabulary = FIXED_VOCABULARY
    print("In fixed vocabulary mode. Training embeddings from scratch.")
    initial_embeddings = None
    # Trim dataset, convert token sequences to integer sequences, crop, and
    # pad.
    print("Preprocessing training data.")
    training_data = PreprocessDataset(
        raw_training_data,
        vocabulary,
        args.seq_len, #def to 100
        eval_mode=False,
        sentence_pair_data=SENTENCE_PAIR_DATA,
        simple=True,
        allow_cropping=False,
        pad_from_left=True)
    training_data_iter = MakeTrainingIterator(training_data, args.batch_size, args.smart_batching, args.use_peano, sentence_pair_data=SENTENCE_PAIR_DATA)
    training_data_length = len(training_data[0])
    # Preprocess eval sets.
    eval_data = PreprocessDataset(
        raw_eval_data,
        vocabulary,
        args.seq_len_test,
        eval_mode=True,
        sentence_pair_data=SENTENCE_PAIR_DATA,
        simple=True, #for RNNs and shit
        allow_cropping=True,
        pad_from_left=True)
    eval_it = MakeEvalIterator(eval_data, args.batch_size_test, None,
        bucket_eval=True,
        shuffle=False)

    return vocabulary, initial_embeddings, training_data_iter, eval_it, training_data_length, raw_eval_data_copy


def MakeTrainingIterator(
        sources,
        batch_size,
        smart_batches=True,
        use_peano=True,
        sentence_pair_data=True,
        pad_from_left=True):
    # Make an iterator that exposes a dataset as random minibatches.

    def get_key(num_transitions):
        if use_peano and sentence_pair_data:
            prem_len, hyp_len = num_transitions
            key = Peano(prem_len, hyp_len)
            return key
        else:
            if not isinstance(num_transitions, list):
                num_transitions = [num_transitions]
            return max(num_transitions)

    def build_batches():
        dataset_size = len(sources[0])
        order = list(range(dataset_size))
        random.shuffle(order)
        order = np.array(order)

        num_splits = 10  # TODO: Should we be smarter about split size?
        order_limit = len(order) // num_splits * num_splits
        order = order[:order_limit]
        order_splits = np.split(order, num_splits)
        batches = []

        for split in order_splits:
            # Put indices into buckets based on example length.
            keys = []
            for i in split:
                num_transitions = sources[3][i]
                key = get_key(num_transitions)
                keys.append((i, key))
            keys = sorted(keys, key=lambda __key: __key[1])

            # Group indices from buckets into batches, so that
            # examples in each batch have similar length.
            batch = []
            for i, _ in keys:
                batch.append(i)
                if len(batch) == batch_size:
                    batches.append(batch)
                    batch = []
        return batches

    def batch_iter():
        batches = build_batches()
        num_batches = len(batches)
        idx = -1
        order = list(range(num_batches))
        random.shuffle(order)

        while True:
            idx += 1
            if idx >= num_batches:
                # Start another epoch.
                batches = build_batches()
                num_batches = len(batches)
                idx = 0
                order = list(range(num_batches))
                random.shuffle(order)
            batch_indices = batches[order[idx]]
            yield tuple(source[batch_indices] for source in sources), num_batches

    def data_iter():
        dataset_size = len(sources[0])
        start = -1 * batch_size
        order = list(range(dataset_size))
        random.shuffle(order)

        while True:
            start += batch_size
            if start > dataset_size - batch_size:
                # Start another epoch.
                start = 0
                random.shuffle(order)
            batch_indices = order[start:start + batch_size]
            yield tuple(source[batch_indices] for source in sources)

    train_iter = batch_iter if smart_batches else data_iter

    return train_iter()

def MakeBucketEvalIterator(sources, batch_size):
    # Order in eval should not matter. Use batches sorted by length for speed
    # improvement.

    def single_sentence_key(num_transitions):
        return num_transitions

    def sentence_pair_key(num_transitions):
        sent1_len, sent2_len = num_transitions
        return Peano(sent1_len, sent2_len)

    dataset_size = len(sources[0])

    # Sort examples by length. From longest to shortest.
    num_transitions = sources[3]
    sort_key = sentence_pair_key if len(
        num_transitions.shape) == 2 else single_sentence_key
    order = sorted(zip(list(range(dataset_size)), num_transitions),
                   key=lambda x: sort_key(x[1]))
    order = list(reversed(order))
    order = [x[0] for x in order]

    num_batches = dataset_size // batch_size
    batches = []

    # Roll examples into batches so they have similar length.
    for i in range(num_batches):
        batch_indices = order[i * batch_size:(i + 1) * batch_size]
        batch = tuple(source[batch_indices] for source in sources)
        batches.append(batch)

    examples_leftover = dataset_size - num_batches * batch_size

    # Create a short batch:
    if examples_leftover > 0:
        batch_indices = order[num_batches *
                              batch_size:num_batches *
                              batch_size +
                              examples_leftover]
        batch = tuple(source[batch_indices] for source in sources)
        batches.append(batch)

    return batches


def MakeEvalIterator(sources, batch_size, limit=None, shuffle=False, rseed=123, bucket_eval=False):
    return MakeBucketEvalIterator(sources, batch_size)[:limit]

def TrimDataset(dataset, seq_length, eval_mode=False,
                sentence_pair_data=False, logger=None, allow_cropping=False):
    """Avoid using excessively long training examples."""

    if sentence_pair_data:
        trimmed_dataset = [
            example for example in dataset if len(
                example["premise_transitions"]) <= seq_length and len(
                example["hypothesis_transitions"]) <= seq_length]
    else:
        trimmed_dataset = [example for example in dataset if
                           len(example["transitions"]) <= seq_length]

    diff = len(dataset) - len(trimmed_dataset)
    if eval_mode:
        assert allow_cropping or diff == 0, "allow_eval_cropping is false but there are over-length eval examples."
        if logger and diff > 0:
            logger.Log(
                "Warning: Cropping " +
                str(diff) +
                " over-length eval examples.")
        return dataset
    else:
        if allow_cropping:
            if logger and diff > 0:
                logger.Log(
                    "Cropping " +
                    str(diff) +
                    " over-length training examples.")
            return dataset
        else:
            if logger and diff > 0:
                logger.Log(
                    "Discarding " +
                    str(diff) +
                    " over-length training examples.")
            return trimmed_dataset

def TokensToIDs(vocabulary, dataset, sentence_pair_data=False):
    """Replace strings in original boolean dataset with token IDs."""
    if sentence_pair_data:
        keys = ["premise_tokens", "hypothesis_tokens"]
    else:
        keys = ["tokens"]

    tokens = 0
    unks = 0
    lowers = 0
    raises = 0

    for key in keys:
        if UNK_TOKEN in vocabulary:
            unk_id = vocabulary[UNK_TOKEN]
            for example in dataset:
                for i, token in enumerate(example[key]):
                    if token in vocabulary:
                        example[key][i] = vocabulary[token]
                    elif token.lower() in vocabulary:
                        example[key][i] = vocabulary[token.lower()]
                        lowers += 1
                    elif token.upper() in vocabulary:
                        example[key][i] = vocabulary[token.upper()]
                        raises += 1
                    else:
                        example[key][i] = unk_id
                        unks += 1
                    tokens += 1
            print("Unk rate {:2.6f}%, downcase rate {:2.6f}%, upcase rate {:2.6f}%".format((unks * 100.0 / tokens), (lowers * 100.0 / tokens), (raises * 100.0 / tokens)))
        else:
            for example in dataset:
                example[key] = [vocabulary[token]
                                for token in example[key]]
    return dataset

def CropAndPadExample(
        example,
        padding_amount,
        target_length,
        key,
        symbol=0,
        logger=None,
        allow_cropping=False,
        pad_from_left=True):
    """
    Crop/pad a sequence value of the given dict `example`.
    """
    if padding_amount < 0:
        if not allow_cropping:
            raise NotImplementedError(
                "Cropping not allowed. "
                "Please set seq_length and eval_seq_length to some sufficiently large value or (for non-SPINN models) use --allow_cropping and --allow_eval_cropping..")
        # Crop, then pad normally.
        if pad_from_left:
            example[key] = example[key][-padding_amount:]
        else:
            example[key] = example[key][:padding_amount]
        padding_amount = 0
    alternate_side_padding = target_length - \
        (padding_amount + len(example[key]))
    if pad_from_left:
        example[key] = ([symbol] * padding_amount) + \
            example[key] + ([symbol] * alternate_side_padding)
    else:
        example[key] = ([symbol] * alternate_side_padding) + \
            example[key] + ([symbol] * padding_amount)


def CropAndPadSimple(
        dataset,
        length,
        logger=None,
        sentence_pair_data=False,
        allow_cropping=True,
        pad_from_left=True):
    # NOTE: This can probably be done faster in NumPy if it winds up making a
    # difference.
    if sentence_pair_data:
        keys = ["premise_tokens",
                "hypothesis_tokens"]
    else:
        keys = ["tokens"]

    for example in dataset:
        for tokens_key in keys:
            num_tokens = len(example[tokens_key])
            tokens_padding_amount = length - num_tokens
            CropAndPadExample(
                example,
                tokens_padding_amount,
                length,
                tokens_key,
                symbol=SENTENCE_PADDING_SYMBOL,
                logger=logger,
                allow_cropping=allow_cropping,
                pad_from_left=pad_from_left)
    return dataset

def truncate(data, seq_length, max_length, left_padded):
    if left_padded:
        data = data[:, seq_length - max_length:]
    else:
        data = data[:, :max_length]
    return data


def get_batch(batch):
    X_batch, transitions_batch, y_batch, num_transitions_batch, example_ids = batch

    # Truncate each batch to max length within the batch.
    X_batch_is_left_padded = True
    transitions_batch_is_left_padded = True
    max_length = np.max(num_transitions_batch)
    seq_length = X_batch.shape[1]

    # Truncate batch.
    X_batch = truncate(X_batch, seq_length, max_length, X_batch_is_left_padded)
    transitions_batch = truncate(transitions_batch, seq_length,
                                 max_length, transitions_batch_is_left_padded)

    return X_batch, transitions_batch, y_batch, num_transitions_batch, example_ids
