import random
import pickle
import numpy as np
from collections import Counter



def load_dictionary(dict_file):
    idx2word = pickle.load(open(dict_file, 'rb'))
    idx2word += ['<unk>']
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return idx2word, word2idx


def stream(data_file, word2idx):
    unk_idx = word2idx['<unk>']
    for line in open(data_file):
        line = line.strip()
        idxs = np.array(
            [word2idx.get(w, unk_idx) for w in line.split()] +
            [word2idx['<eos>']],
            dtype=np.int32
        )
        yield idxs


def randomise(stream, buffer_size=100):
    buf = buffer_size * [None]
    ptr = 0
    for item in stream:
        buf[ptr] = item
        ptr += 1
        if ptr == buffer_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            ptr = 0
    buf = buf[:ptr]
    random.shuffle(buf)
    for x in buf:
        yield x


def sortify(stream, key, buffer_size=200):
    buf = buffer_size * [None]
    ptr = 0
    for item in stream:
        buf[ptr] = item
        ptr += 1
        if ptr == buffer_size:
            buf.sort(key=key)
            for x in buf:
                yield x
            ptr = 0
    #buf = buf[:ptr]
    #buf.sort(key=key)

    for x in sorted(buf[:ptr], key=key):
        yield x


def batch(stream, batch_size=10, ignore_final=False):
    batch = []
    for item in stream:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if not ignore_final and len(batch) > 0:
        yield batch


def arrayify(stream, pad):
    for batch in stream:
        word_dim = max(len(s) for s in batch)
        batch_idxs = np.full((len(batch), word_dim), pad, dtype=np.int32)
        for i in range(len(batch)):
            sentence = batch[i]
            batch_idxs[i, :len(sentence)] = sentence
        yield batch_idxs

if __name__ == "__main__":
    import sys
    data_file = sys.argv[1]
    pickle.dump(create_dictionary(data_file, top_k=10000),
                open('dict.pkl', 'wb'))
