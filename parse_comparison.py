"""
Reads a parsed corpus (data_path) and a model report (report_path) from a model
that produces latent tree structures and computes the unlabeled F1 score between
the model's latent trees and:
- The ground-truth trees in the parsed corpus
- Strictly left-branching trees for the sentences in the parsed corpus
- Strictly right-branching trees for the sentences in the parsed corpus

Note that for binary-branching trees like these, precision, recall, and F1 are
equal by definition, so only one number is shown.

Usage:
$ python scripts/parse_comparison.py \
    --data_path ./snli_1.0/snli_1.0_dev.jsonl \
    --report_path ./logs/example-nli.report \
"""

import gflags
import sys
import codecs
import json
import random
import re
import glob
import math
from collections import Counter

LABEL_MAP = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

FLAGS = gflags.FLAGS

mathops = ["[MAX", "[MIN", "[MED", "[SM"]


def spaceify(parse):
    return parse  # .replace("(", "( ").replace(")", " )")


def balance(parse, lowercase=False):
    # Modified to provided a "half-full" binary tree without padding.
    # Difference between the other method is the right subtrees are
    # the half full ones.
    tokens = tokenize_parse(parse)
    if len(tokens) > 1:
        transitions = full_transitions(len(tokens), right_full=True)
        stack = []
        for transition in transitions:
            if transition == 0:
                stack.append(tokens.pop(0))
            elif transition == 1:
                right = stack.pop()
                left = stack.pop()
                stack.append("( " + left + " " + right + " )")
        assert len(stack) == 1
    else:
        stack = tokens
    return stack[0]


def roundup2(N):
    """ Round up using factors of 2. """
    return int(2 ** math.ceil(math.log(N, 2)))


def full_transitions(N, left_full=False, right_full=False):
    """
    Recursively creates a full binary tree of with N
    leaves using shift reduce transitions.
    """

    if N == 1:
        return [0]

    if N == 2:
        return [0, 0, 1]

    assert not (left_full and right_full), "Please only choose one."

    if not left_full and not right_full:
        N = float(N)

        # Constrain to full binary trees.
        assert math.log(N, 2) % 1 == 0, \
            "Bad value. N={}".format(N)

        left_N = N / 2
        right_N = N - left_N

    if left_full:
        left_N = roundup2(N) / 2
        right_N = N - left_N

    if right_full:
        right_N = roundup2(N) / 2
        left_N = N - right_N

    return full_transitions(left_N, left_full=left_full, right_full=right_full) + \
           full_transitions(right_N, left_full=left_full, right_full=right_full) + \
           [1]


def tokenize_parse(parse):
    parse = spaceify(parse)
    return [token for token in parse.split() if token not in ['(', ')']]


def to_string(parse):
    if type(parse) is not list:
        return parse
    if len(parse) == 1:
        return parse[0]
    else:
        return '( ' + to_string(parse[0]) + ' ' + to_string(parse[1]) + ' )'


def tokens_to_rb(tree):
    if type(tree) is not list:
        return tree
    if len(tree) == 1:
        return tree[0]
    else:
        return [tree[0], tokens_to_rb(tree[1:])]


def to_rb(gt_table):
    new_data = {}
    for key in gt_table:
        parse = gt_table[key]
        tokens = tokenize_parse(parse)
        new_data[key] = to_string(tokens_to_rb(tokens))
    return new_data


def tokens_to_lb(tree):
    if type(tree) is not list:
        return tree
    if len(tree) == 1:
        return tree[0]
    else:
        return [tokens_to_lb(tree[:-1]), tree[-1]]


def to_lb(gt_table):
    new_data = {}
    for key in gt_table:
        parse = gt_table[key]
        tokens = tokenize_parse(parse)
        new_data[key] = to_string(tokens_to_lb(tokens))
    return new_data


def average_depth(parse):
    depths = []
    current_depth = 0
    for token in parse.split():
        if token == '(':
            current_depth += 1
        elif token == ')':
            current_depth -= 1
        else:
            depths.append(current_depth)
    if len(depths) == 0:
        pass
    else:
        return float(sum(depths)) / len(depths)


def corpus_average_depth(corpus):
    local_averages = []
    for key in corpus:
        s = corpus[key]
        if average_depth(s) is not None:
            local_averages.append(average_depth(s))
        else:
            pass
    return float(sum(local_averages)) / len(local_averages)


def average_length(parse):
    parse = spaceify(parse)
    return len(parse.split())


def corpus_average_length(corpus):
    local_averages = []
    for key in corpus:
        if average_length(s) is not None:
            local_averages.append(average_length(s))
        else:
            pass
    return float(sum(local_averages)) / len(local_averages)


def corpus_stats(corpus_1, corpus_2, first_two=False, neg_pair=False, const_parse=False):
    """
    Note: If a few examples in one dataset are missing from the other (i.e., some examples from the source corpus were not included
      in a model corpus), the shorter dataset must be supplied as corpus_1.

    corpus_1 is the report being evaluated (important for counting complete constituents)
    """

    f1_accum = 0.0
    count = 0.0
    first_two_count = 0.0
    last_two_count = 0.0
    three_count = 0.0
    neg_pair_count = 0.0
    neg_count = 0.0
    const_parsed_1 = 0
    if const_parse:
        const_parsed_2 = 0
    else:
        const_parsed_2 = 1
    for key in corpus_2:
        c1, cp1 = to_indexed_contituents(corpus_1[key], const_parse)
        c2, cp2 = to_indexed_contituents(corpus_2[key], const_parse)
        f1_accum += example_f1(c1, c2)
        count += 1
        const_parsed_1 += cp1
        const_parsed_2 += cp2

        if first_two and len(c1) > 1:
            if (0, 2) in c1:
                first_two_count += 1
            num_words = len(c1) + 1
            if (num_words - 2, num_words) in c1:
                last_two_count += 1
            three_count += 1
        if neg_pair:
            word_index = 0
            s = spaceify(corpus_1[key])
            tokens = s.split()
            for token_index, token in enumerate(tokens):
                if token in ['(', ')']:
                    continue
                if token in ["n't", "not", "never", "no", "none", "Not", "Never", "No", "None"]:
                    if tokens[token_index + 1] not in ['(', ')']:
                        neg_pair_count += 1
                    neg_count += 1
                word_index += 1
    stats = f1_accum / count
    if first_two:
        stats = str(stats) + '\t' + str(first_two_count / three_count) + '\t' + str(last_two_count / three_count)
    if neg_pair:
        stats = str(stats) + '\t' + str(neg_pair_count / neg_count)
    return stats, const_parsed_1 / const_parsed_2


def corpus_stats_labeled(corpus_unlabeled, corpus_labeled):
    """
    Note: If a few examples in one dataset are missing from the other (i.e., some examples from the source corpus were not included
      in a model corpus), the shorter dataset must be supplied as corpus_1.
    """

    correct = Counter()
    total = Counter()

    for key in corpus_labeled:
        c1, _, nwords1 = to_indexed_contituents(corpus_unlabeled[key], False)
        c2, nwords2 = to_indexed_contituents_labeled(corpus_labeled[key])
        assert nwords1 == nwords2
        if len(c2) == 0:
            continue

        ex_correct, ex_total = example_labeled_acc(c1, c2)
        correct.update(ex_correct)
        total.update(ex_total)
    return correct, total


def count_parse(parse, index, const_parsed=[]):
    """
    Compute Constituents Parsed metric for ListOps style examples.
    """
    mathops = ["[MAX", "[MIN", "[MED", "[SM"]
    if "]" in parse:
        after = parse[index:]
        before = parse[:index]
        between = after[: after.index("]")]

        nest_check = [m in between[1:] for m in mathops]
        if True in nest_check:
            op_i = nest_check.index(True)
            nested_i = after[1:].index(mathops[op_i]) + 1
            nested = after[nested_i:]
            c = count_parse(parse, index + nested_i, const_parsed)
            cc = count_parse(parse, index, const_parsed)
        else:
            o_b = between.count("(")  # open, between
            c_b = between.count(")")  # close, between

            end = after.index("]")
            cafter = after[end + 1:]
            stop = None
            stop_list = []
            for item in cafter:
                stop_list.append(")" == item)
                if stop_list[-1] == False:
                    break

            if False in stop_list:
                stop = stop_list.index(False)
            else:
                stop = None
            cafter = cafter[: stop]
            c_a = cafter.count(")")

            stop = None
            stop_list = []
            for item in before[::-1]:
                stop_list.append("(" == item)
                if stop_list[-1] == False:
                    break

            if False in stop_list:
                stop = len(before) - stop_list.index(False) - 1
            else:
                stop = None
            cbefore = before[stop:]
            o_a = cbefore.count("(")

            ints = sum(c.isdigit() for c in between) + between.count("-")
            op = o_a + o_b
            cl = c_a + c_b

            if op >= ints and cl >= ints:
                if op == ints + 1 or cl == ints + 1:
                    const_parsed.append(1)
            parse[index - o_a: index + len(between) + 1 + c_a] = '-'
    return sum(const_parsed)


def to_indexed_contituents(parse, const_parse):
    if parse.count("(") != parse.count(")"):
        print(parse)
    parse = spaceify(parse)
    sp = parse.split()
    if len(sp) == 1:
        return set([(0, 1)]), 0, 1

    backpointers = []
    indexed_constituents = set()
    word_index = 0
    first_op = -1
    for index, token in enumerate(sp):
        if token == '(':
            backpointers.append(word_index)
        elif token == ')':
            # if len(backpointers) == 0:
            #    pass
            # else:
            start = backpointers.pop()
            end = word_index
            constituent = (start, end)
            indexed_constituents.add(constituent)
        elif "[" in token:
            if first_op == -1:
                first_op = index
            else:
                pass
        else:
            word_index += 1

    const_parsed = []
    cp = 0
    if const_parse:
        cp = count_parse(sp, first_op, const_parsed)
        max_count = parse.count("]")
    return indexed_constituents, cp, word_index


def to_indexed_contituents_labeled(parse):
    # sp = re.findall(r'\([^ ]+| [^\(\) ]+|\)', parse)
    sp = parse.split()
    if len(sp) == 1:
        return set([(0, 1)])

    backpointers = []
    indexed_constituents = set()
    word_index = 0
    for index, token in enumerate(sp):
        if token[0] == '(':
            backpointers.append((word_index, token[1:]))
        elif token == ')':
            start, typ = backpointers.pop()
            end = word_index
            constituent = (start, end, typ)
            if end - start > 1:
                indexed_constituents.add(constituent)
        else:
            word_index += 1
    return indexed_constituents, word_index


def example_f1(c1, c2):
    prec = float(len(c1.intersection(c2))) / len(c2)
    return prec  # For strictly binary trees, P = R = F1


def example_labeled_acc(c1, c2):
    '''Compute the number of non-unary constituents of each type in the labeled (non-binirized) parse appear in the model output.'''
    correct = Counter()
    total = Counter()
    for constituent in c2:
        if (constituent[0], constituent[1]) in c1:
            correct[constituent[2]] += 1
        total[constituent[2]] += 1
    return correct, total


def randomize(parse):
    tokens = tokenize_parse(parse)
    while len(tokens) > 1:
        merge = random.choice(list(range(len(tokens) - 1)))
        tokens[merge] = "( " + tokens[merge] + " " + tokens[merge + 1] + " )"
        del tokens[merge + 1]
    return tokens[0]


def to_latex(parse):
    return ("\\Tree " + parse).replace('(', '[').replace(')', ']').replace(' . ', ' $.$ ')


def read_nli_report(path):
    report = {}
    with codecs.open(path, encoding='utf-8') as f:
        for line in f:
            loaded_example = json.loads(line)
            report[loaded_example['example_id'] + "_1"] = unpad(loaded_example['sent1_tree'])
            report[loaded_example['example_id'] + "_2"] = unpad(loaded_example['sent2_tree'])
    return report


def read_sst_report(path):
    report = {}
    with codecs.open(path, encoding='utf-8') as f:
        for line in f:
            loaded_example = json.loads(line)
            report[loaded_example['example_id'] + "_1"] = unpad(loaded_example['sent1_tree'])
    return report


def read_listops_report(path):
    report = {}
    correct = 0
    num = 0
    with codecs.open(path, encoding='utf-8') as f:
        for line in f:
            loaded_example = json.loads(line)
            report[loaded_example['example_id']] = unpad(loaded_example['sent1_tree'])
            num += 1
            if loaded_example['truth'] == loaded_example['prediction']:
                correct += 1
    print("Accuracy = ", correct / num)
    return report


def read_nli_report_padded(path):
    report = {}
    with codecs.open(path, encoding='utf-8') as f:
        for line in f:
            try:
                line = line.encode('UTF-8')
            except UnicodeError as e:
                print("ENCODING ERROR:", line, e)
                line = "{}"
            loaded_example = json.loads(line)
            report[loaded_example['example_id'] + "_1"] = loaded_example['sent1_tree']
            report[loaded_example['example_id'] + "_2"] = loaded_example['sent2_tree']
    return report


def read_ptb_report(path):
    report = {}
    with codecs.open(path, encoding='utf-8') as f:
        for line in f:
            loaded_example = json.loads(line)
            report[loaded_example['example_id']] = unpad(loaded_example['sent1_tree'])
    return report


def unpad(parse):
    ok = ["(", ")", "_PAD"]
    unpadded = []
    tokens = parse.split()
    cur = [i for i in range(len(tokens)) if tokens[i] == "_PAD"]

    if len(cur) != 0:
        if tokens[cur[0] - 1] in ok:
            unpad = tokens[:cur[0] - 1]
        else:
            unpad = tokens[:cur[0]]
    else:
        unpad = tokens

    sent = " ".join(unpad)
    while sent.count("(") != sent.count(")"):
        sent += " )"

    return sent


def ConvertBinaryBracketedSeq(seq):
    T_SHIFT = 0
    T_REDUCE = 1

    tokens, transitions = [], []
    for item in seq:
        if item != "(":
            if item != ")":
                tokens.append(item)
            transitions.append(T_REDUCE if item == ")" else T_SHIFT)
    return tokens, transitions


def run():
    gt = {}
    # gt_labeled = {}
    with codecs.open(FLAGS.main_data_path, encoding='utf-8') as f:
        for example_id, line in enumerate(f):
            if FLAGS.data_type == "nli":
                loaded_example = json.loads(line)
                if loaded_example["gold_label"] not in LABEL_MAP:
                    continue
                if '512-4841' in loaded_example['sentence1_binary_parse'] \
                        or '512-8581' in loaded_example['sentence1_binary_parse'] \
                        or '412-4841' in loaded_example['sentence1_binary_parse'] \
                        or '512-4841' in loaded_example['sentence2_binary_parse'] \
                        or '512-8581' in loaded_example['sentence2_binary_parse'] \
                        or '412-4841' in loaded_example['sentence2_binary_parse']:
                    continue  # Stanford parser tree binarizer doesn't handle phone numbers properly.
                gt[loaded_example['pairID'] + "_1"] = loaded_example['sentence1_binary_parse']
                gt[loaded_example['pairID'] + "_2"] = loaded_example['sentence2_binary_parse']
                # gt_labeled[loaded_example['pairID'] + "_1"] = loaded_example['sentence1_parse']
                # gt_labeled[loaded_example['pairID'] + "_2"] = loaded_example['sentence2_parse']

                gt_labeled[loaded_example['pairID'] + "_1"] = loaded_example['sentence1_parse']
                gt_labeled[loaded_example['pairID'] + "_2"] = loaded_example['sentence2_parse']

            elif FLAGS.data_type == "sst":
                line = line.strip()
                stack = []
                words = line.replace(')', ' )')
                words = words.split(' ')
                for index, word in enumerate(words):
                    if word[0] != "(":
                        if word == ")":
                            # Ignore unary merges
                            if words[index - 1] == ")":
                                newg = "( " + stack.pop() + " " + stack.pop() + " )"
                                stack.append(newg)
                        else:
                            stack.append(word)
                gt[str(example_id) + "_1"] = stack[0]

            elif FLAGS.data_type == "listops":
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
                gt[example["example_id"]] = example["sentence"]

    lb = to_lb(gt)
    rb = to_rb(gt)
    print("GT average depth", corpus_average_depth(gt))

    ptb = {}
    ptb_labeled = {}
    if FLAGS.ptb_data_path != "_":
        with codecs.open(FLAGS.ptb_data_path, encoding='utf-8') as f:
            for line in f:
                loaded_example = json.loads(line)
                if loaded_example["gold_label"] not in LABEL_MAP:
                    continue
                ptb[loaded_example['pairID']] = loaded_example['sentence1_binary_parse']
                ptb_labeled[loaded_example['pairID']] = loaded_example['sentence1_parse']

    reports = []
    ptb_reports = []
    if FLAGS.use_random_parses:
        print("Creating five sets of random parses for the main data.")
        report_paths = list(range(5))
        for _ in report_paths:
            report = {}
            for sentence in gt:
                report[sentence] = randomize(gt[sentence])
            reports.append(report)

        print("Creating five sets of random parses for the PTB data.")
        ptb_report_paths = list(range(5))
        for _ in report_paths:
            report = {}
            for sentence in ptb:
                report[sentence] = randomize(ptb[sentence])
            ptb_reports.append(report)
    if FLAGS.use_balanced_parses:
        print("Creating five sets of balanced binary parses for the main data.")
        report_paths = list(range(5))
        for _ in report_paths:
            report = {}
            for sentence in gt:
                report[sentence] = balance(gt[sentence])
            reports.append(report)

        print("Creating five sets of balanced binary parses for the PTB data.")
        ptb_report_paths = list(range(5))
        for _ in report_paths:
            report = {}
            for sentence in ptb:
                report[sentence] = balance(ptb[sentence])
            ptb_reports.append(report)
    else:
        report_paths = glob.glob(FLAGS.main_report_path_template)
        for path in report_paths:
            print("Loading", path)
            if FLAGS.data_type == "nli":
                reports.append(read_nli_report(path))
            elif FLAGS.data_type == "sst":
                reports.append(read_sst_report(path))
            elif FLAGS.data_type == "listops":
                reports.append(read_listops_report(path))
        if FLAGS.main_report_path_template != "_":
            ptb_report_paths = glob.glob(FLAGS.ptb_report_path_template)
            for path in ptb_report_paths:
                print("Loading", path)
                ptb_reports.append(read_ptb_report(path))

    if len(reports) > 1 and FLAGS.compute_self_f1:
        f1s = []
        for i in range(len(report_paths) - 1):
            for j in range(i + 1, len(report_paths)):
                path_1 = report_paths[i]
                path_2 = report_paths[j]
                f1 = corpus_stats(reports[i], reports[j])
                f1s.append(f1)
        print("Mean Self F1:\t" + str(sum(f1s) / len(f1s)))

    correct = Counter()
    total = Counter()
    for i, report in enumerate(reports):
        print(report_paths[i])
        if FLAGS.print_latex > 0:
            for index, sentence in enumerate(gt):
                if index == FLAGS.print_latex:
                    break
                print(to_latex(gt[sentence]))
                print(to_latex(report[sentence]))
                print()

        if FLAGS.data_type == "listops":
            gtf1, gtcp = corpus_stats(report, gt, first_two=FLAGS.first_two, neg_pair=FLAGS.neg_pair, const_parse=True)
        else:
            gtf1, gtcp = corpus_stats(report, gt, first_two=FLAGS.first_two, neg_pair=FLAGS.neg_pair, const_parse=False)
        print("Left:", str(corpus_stats(report, lb)[0]) + '\t' + "Right:",
              str(corpus_stats(report, rb)[0]) + '\t' + "Groud-truth", str(gtf1) + '\t' + "Tree depth:",
              str(corpus_average_depth(report)), '\t', "Constituent parsed:", str(gtcp))

    correct = Counter()
    total = Counter()
    for i, report in enumerate(ptb_reports):
        print(ptb_report_paths[i])
        if FLAGS.print_latex > 0:
            for index, sentence in enumerate(ptb):
                if index == FLAGS.print_latex:
                    break
                print(to_latex(ptb[sentence]))
                print(to_latex(report[sentence]))
                print()
        print(str(corpus_stats(report, ptb)) + '\t' + str(corpus_average_depth(report)))
        set_correct, set_total = corpus_stats_labeled(report, ptb_labeled)
        correct.update(set_correct)
        total.update(set_total)

    for key in sorted(total):
        print(key + '\t' + str(correct[key] * 1. / total[key]))


if __name__ == '__main__':
    gflags.DEFINE_string("main_report_path_template", "./checkpoints/*.report",
                         "A template (with wildcards input as \*) for the paths to the main reports.")
    gflags.DEFINE_string("main_data_path", "./snli_1.0/snli_1.0_dev.jsonl",
                         "A template (with wildcards input as \*) for the paths to the main reports.")
    gflags.DEFINE_string("ptb_report_path_template", "_",
                         "A template (with wildcards input as \*) for the paths to the PTB reports, or '_' if not available.")
    gflags.DEFINE_string("ptb_data_path", "_", "The path to the PTB data in SNLI format, or '_' if not available.")
    gflags.DEFINE_boolean("compute_self_f1", True,
                          "Compute self F1 over all reports matching main_report_path_template.")
    gflags.DEFINE_boolean("use_random_parses", False,
                          "Replace all report trees with randomly generated trees. Report path template flags are not used when this is set.")
    gflags.DEFINE_boolean("use_balanced_parses", False,
                          "Replace all report trees with roughly-balanced binary trees. Report path template flags are not used when this is set.")
    gflags.DEFINE_boolean("first_two", False, "Show 'first two' and 'last two' metrics.")
    gflags.DEFINE_boolean("neg_pair", False, "Show 'neg_pair' metric.")
    gflags.DEFINE_enum("data_type", "nli", ["nli", "sst", "listops"], "Data Type")
    gflags.DEFINE_integer("print_latex", 0, "Print this many trees in LaTeX format for each report.")

    FLAGS(sys.argv)

    run()