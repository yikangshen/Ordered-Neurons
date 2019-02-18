import torch


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def load_embeddings_txt(path):
  words = pd.read_csv(path, sep=" ", index_col=0,
                      na_values=None, keep_default_na=False, header=None,
                      quoting=csv.QUOTE_NONE)
  matrix = words.values
  index_to_word = list(words.index)
  word_to_index = {
    word: ind for ind, word in enumerate(index_to_word)
  }
  return matrix, word_to_index, index_to_word

def evalb(pred_tree_list, targ_tree_list):
    import os
    import subprocess
    import tempfile
    import re
    import nltk

    temp_path = tempfile.TemporaryDirectory(prefix="evalb-")
    temp_file_path = os.path.join(temp_path.name, "pred_trees.txt")
    temp_targ_path = os.path.join(temp_path.name, "true_trees.txt")
    temp_eval_path = os.path.join(temp_path.name, "evals.txt")

    print("Temp: {}, {}".format(temp_file_path, temp_targ_path))
    temp_tree_file = open(temp_file_path, "w")
    temp_targ_file = open(temp_targ_path, "w")

    for pred_tree, targ_tree in zip(pred_tree_list, targ_tree_list):
        def process_str_tree(str_tree):
            return re.sub('[ |\n]+', ' ', str_tree)

        def list2tree(node):
            if isinstance(node, list):
                tree = []
                for child in node:
                    tree.append(list2tree(child))
                return nltk.Tree('<unk>', tree)
            elif isinstance(node, str):
                return nltk.Tree('<word>', [node])

        temp_tree_file.write(process_str_tree(str(list2tree(pred_tree)).lower()) + '\n')
        temp_targ_file.write(process_str_tree(str(list2tree(targ_tree)).lower()) + '\n')

    temp_tree_file.close()
    temp_targ_file.close()

    evalb_dir = os.path.join(os.getcwd(), "EVALB")
    evalb_param_path = os.path.join(evalb_dir, "COLLINS.prm")
    evalb_program_path = os.path.join(evalb_dir, "evalb")
    command = "{} -p {} {} {} > {}".format(
        evalb_program_path,
        evalb_param_path,
        temp_targ_path,
        temp_file_path,
        temp_eval_path)

    subprocess.run(command, shell=True)

    with open(temp_eval_path) as infile:
        for line in infile:
            match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
            if match:
                evalb_recall = float(match.group(1))
            match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
            if match:
                evalb_precision = float(match.group(1))
            match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
            if match:
                evalb_fscore = float(match.group(1))
                break

    temp_path.cleanup()

    print('-' * 80)
    print('Evalb Prec:', evalb_precision,
          ', Evalb Reca:', evalb_recall,
          ', Evalb F1:', evalb_fscore)

    return evalb_fscore
