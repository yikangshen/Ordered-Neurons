import sys
import numpy as np
import pickle
import argparse
from template.TestCases import TestCase

parser = argparse.ArgumentParser(description="Parameters for sampling from the templates")

parser.add_argument('--num', type=int, default=10,
                    help='Number of samples to generate for each construction.')
parser.add_argument('--template_dir', type=str, default='../EMNLP2018/templates',
                    help='Location of the template files')
parser.add_argument('--format', type=str, default='txt',
                    help='Format to save the samples (txt/csv)')
parser.add_argument('--out_file', type=str, default='human_test_sents',
                    help='File to output the resulting sentences.')
args = parser.parse_args()

testcases=TestCase().all_cases
fout = open(args.out_file+'.'+args.format, 'w')
if args.format == "csv":
    sep=","
else: sep="\t"

fout.write("test case"+sep+"grammatical"+sep+"ungrammatical\n")

for t in testcases:
    templates = pickle.load(open(args.template_dir+"/"+t+'.pickle', 'rb'))
    if t != "simple_agrmt":
        for key in templates.keys():
            all_sents = ['_vs_'.join(x) for x in templates[key]]
            sents = np.random.choice(all_sents, args.num)
            for i in range(args.num):
                pair = sents[i].split("_vs_")
                fout.write(t+"--"+key+sep+pair[0]+"."+sep+pair[1]+".\n")
    else:
        # We have 5x as many simple_agrmt cases as an 'attention check'
        for key in templates.keys():
            all_sents = ['_vs_'.join(x) for x in templates[key]]
            for i in range(5*args.num):
                sents = np.random.choice(all_sents, args.num)
                for j in range(args.num):
                    pair = sents[j].split("_vs_")
                    fout.write(t+"--"+key+"--"+str(i)+sep+pair[0]+"."+sep+pair[1]+".\n")
        
fout.close()
