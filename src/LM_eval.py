import argparse
import pickle
import os
import subprocess
import operator
import logging
from progress.bar import Bar
from tester.TestWriter import TestWriter
from template.TestCases import TestCase

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Parameters for testing a language model")

parser.add_argument('--template_dir', type=str, default='./templates',
                    help='Location of the template files')
parser.add_argument('--output_file', type=str, default='all_test_sents.txt',
                    help='File to store all of the sentences that will be tested')
parser.add_argument('--model', type=str, default='./models/lstm_lm.pt',
                    help='The model to test')
parser.add_argument('--lm_data', type=str, default='./models/lstm_lm.bin',
                    help='The model .bin file that accompanies the model (for faster loading)')
parser.add_argument('--tests', type=str, default='all',
                    help='Which constructions to test (agrmt/npi/all)')
parser.add_argument('--model_type', type=str, required=True,
                    help='Which kind of model (RNN/multitask/ngram)')
parser.add_argument('--unit_type', type=str, default='word',
                    help='Kinds of units used on language model (word/char)')
parser.add_argument('--ngram_order', type=int, default=5,
                    help='Order of the ngram model')
parser.add_argument('--vocab', type=str, default='ngram_vocab.pkl',
                    help='File containing the ngram vocab')

args = parser.parse_args()

writer = TestWriter(args.template_dir, args.output_file)
testcase = TestCase()
if args.tests == 'agrmt':
    tests = testcase.agrmt_cases
elif args.tests == 'npi':
    tests = testcase.npi_cases
else:
    tests = testcase.all_cases

all_test_sents = {}
for test_name in tests:
    test_sents = pickle.load(open(args.template_dir+"/"+test_name+".pickle", 'rb'))
    all_test_sents[test_name] = test_sents

writer.write_tests(all_test_sents, args.unit_type)
name_lengths = writer.name_lengths
key_lengths = writer.key_lengths

def test_LM():
    if args.model_type.lower() == "ngram":
        logging.info("Testing ngram...")
        os.system('ngram -order ' + str(args.ngram_order) + ' -lm ' + args.model + ' -vocab ' + args.vocab + ' -ppl ' + args.template_dir+'/'+args.output_file + ' -debug 2 > ngram.output')
        if args.ngram_order == 1:
            results = score_unigram()
        else:
            results = score_ngram()
    else:       
        logging.info("Testing RNN...")
        os.system('./example_scripts/test.sh '+ args.template_dir + ' ' +  args.model + ' ' + args.lm_data + ' ' + args.output_file + ' > '+ 'rnn.output')
        results = score_rnn()
    with open(args.model_type+"_results.pickle", 'wb') as f:
        pickle.dump(results, f)

def score_unigram():
    logging.info("Scoring unigram...")
    fin = open("unigram.output", 'r')
    all_scores = {}
    sent = ""
    prevLineEmpty = True
    i = 0
    for line in fin:
        if "p( " in line:
            word = line.split("p( ")[1].split(" |")[0]
            score = float(line.split("[ ")[-1].split(" ]")[0])
            if word not in all_scores:
                all_scores[word] = score
    fin.close()
    return all_scores

def score_ngram():
    fin = open("ngram.output", 'r')
    all_scores = {}
    i = 0
    finished = True
    sent = []
    prev_sentid = -1
    for line in fin:
        if "p(" in line:
            finished = False
        if not finished and "</s>" not in line:
            word = line.split("p( ")[1].split(" |")[0]
            score = float(line.split("[ ")[-1].split(" ]")[0])
            sent.append((word,score))
            if word == "<eos>":
                name_found = False
                for (k1,v1) in sorted(name_lengths.items(), key=operator.itemgetter(1)):
                    if i < v1 and not name_found:
                        name_found = True
                        if k1 not in all_scores:
                            all_scores[k1] = {}
                        key_found = False
                        for (k2,v2) in sorted(key_lengths[k1].items(), key=operator.itemgetter(1)):
                            if i <  v2 and not key_found:
                                key_found = True
                                if k2 not in all_scores[k1]:
                                    all_scores[k1][k2] = []
                                all_scores[k1][k2].append(sent)
                sent = []
                if i != prev_sentid+1:
                    logging.info("Error at sents "+sentid+" and "+prev_sentid)
                prev_sentid = i
                finished = True
                i += 1
        else:
            finished = True
    fin.close()
    return all_scores


def score_rnn():
    logging.info("Scoring RNN...")
    with open('rnn.output', 'r') as f:
        all_scores = {}
        first = False
        score = 0.
        sent = []
        prev_sentid = -1
        for line in f:
            if line.strip() == "":
                first = True
            elif "===========================" in line:
                first = False
                break
            elif first and len(line.strip().split()) == 6 and "torch.cuda" not in line:
                wrd, sentid, wrd_score = [line.strip().split()[i] for i in [0,1,4]]
                score = -1 * float(wrd_score) # multiply by -1 to turn surps back into logprobs
                sent.append((wrd, score))
                if wrd == ".":
                    name_found = False
                    for (k1,v1) in sorted(name_lengths.items(), key=operator.itemgetter(1)):
                        if float(sentid) < v1 and not name_found:
                            name_found = True
                            if k1 not in all_scores:
                                all_scores[k1] = {}
                            key_found = False
                            for (k2,v2) in sorted(key_lengths[k1].items(), key=operator.itemgetter(1)):
                                if int(sentid) <  v2 and not key_found:
                                    key_found = True
                                    if k2 not in all_scores[k1]:
                                        all_scores[k1][k2] = []
                                    all_scores[k1][k2].append(sent)
                    sent = []
                    if float(sentid) != prev_sentid+1:
                        logging.info("Error at sents "+sentid+" and "+prev_sentid)
                    prev_sentid = float(sentid)
    return all_scores

def clean_files(mode):
    if args.model_type.lower() == 'ngram':
        os.system('rm ngram.output unigram.output')
    else:
        os.system('rm rnn.output')


test_LM()
