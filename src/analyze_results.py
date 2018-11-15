import os
import sys
import pickle
import operator
import logging
import argparse
from template.TestCases import TestCase

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Which kind of results to display")

parser.add_argument('--results_file', type=str, required=True,
                    help='Path to where the results for the LM are stored')
parser.add_argument('--model_type', type=str, required=True,
                    help='Which kind of model (RNN/multitask/ngram)')
parser.add_argument('--analysis', type=str, default='full_sent',
                    help='How to compare scores (full_sent or word_only)')
parser.add_argument('--tests', type=str, default='all',
                    help='Which constructions to test (agrmt/npi/all)')
parser.add_argument('--anim', action='store_true', default=False,
                    help='Examine the effect of animacy on the results')
parser.add_argument('--out_dir', type=str, default='./results',
                    help='Directory to store the results files')
parser.add_argument('--mode', type=str, default='overall',
                    help='Level of detail to report (overall/condensed/full)')
parser.add_argument('--unit_type', type=str, default='word',
                    help='Unit used for language modeling (word/char)')
args = parser.parse_args()
# check args
if args.analysis != 'full_sent' and args.analysis != 'word_only':
    logging.info("ERROR: analysis argument must be 'full_sent' or 'word_only'")
    sys.exit(1)
if args.mode != 'overall' and args.mode != 'condensed' and args.mode != 'full':
    logging.info("ERROR: mode argument must be 'overall' or 'condensed' or 'full'")
    sys.exit(1)

directory=args.out_dir+"/"+args.model_type+"/"+args.analysis
os.system("mkdir -p " + directory)
if os.path.exists(os.path.join(directory, "case_accs.txt")):
    os.system("rm " + os.path.join(directory, "case_accs.txt"))
if os.path.exists(os.path.join(directory, "individual_accs.txt")):
    os.system("rm " + os.path.join(directory, "individual_accs.txt"))
    
testcase = TestCase()
if args.tests == 'agrmt':
    tests = testcase.agrmt_cases
elif args.tests == 'npi':
    tests = testcase.npi_cases
else:
    tests = testcase.all_cases

results = pickle.load(open(args.results_file, 'rb'))

if not args.anim:
    ### JOIN ANIM + INANIM CASES ###
    joined_results = {}
    for name in tests:
        if 'anim' in name:
            new_name = '_'.join(name.split("_")[:-1])
        else:
            new_name = name
        for sub_case in results[name]:
            if new_name not in joined_results:
                joined_results[new_name] = {}
            if sub_case not in joined_results[new_name]:
                joined_results[new_name][sub_case] = []
            joined_results[new_name][sub_case] += results[name][sub_case]
    # dump joined results to .pickle file
    pickle.dump(joined_results, open(args.results_file.split(".pickle")[0]+".joined.pickle", 'wb'))
else:
    joined_results = results

def is_more_probable(sent_a, sent_b):
    if len(sent_a) != len(sent_b) and args.unit_type == 'word':
        logging.info("ERROR: Mismatch in sentence lengths: (1) ",sent_a, " vs (2) ",sent_b)
    if args.analysis == 'word_only':
        index = [sent_a[i][0]!=sent_b[i][0] for i in range(len(sent_a))].index(True)
        return sent_a[index][1] > sent_b[index][1]
    return sum([sent_a[i][1] for i in range(len(sent_a))]) > sum([sent_b[i][1] for i in range(len(sent_b))])

def analyze_agrmt_results(results):
    correct_sents = {}
    incorrect_sents = {}
    for case in results.keys():
        correct_sents[case] = []
        incorrect_sents[case] = []
        for i in range(0,len(results[case]),2):
            grammatical = results[case][i]
            ungrammatical = results[case][i+1]
            if is_more_probable(grammatical, ungrammatical):
                if args.unit_type != 'word':
                    print(grammatical)
                    grammatical = ''.join(''.join(grammatical).split('\s'))
                    ungrammatical = ''.join(''.join(ungrammatical).split('\s'))
                correct_sents[case].append((grammatical, ungrammatical))
            else:
                if args.unit_type != 'word':
                    print(grammatical)
                    grammatical = ''.join(''.join(grammatical).split('\s'))
                    ungrammatical = ''.join(''.join(ungrammatical).split('\s'))
                incorrect_sents[case].append((grammatical, ungrammatical))
    return correct_sents, incorrect_sents

def analyze_npi_results(results):
    options = ['gi_g', 'gi_i','iu_i','iu_u','gu_g','gu_u']
    sentences = {}
    for opt in options:
        sentences[opt] = {}
        for case in results.keys():
            sentences[opt][case] = []
    for case in results.keys():
        for i in range(0,len(results[case]),3):
            grammatical = results[case][i]
            intrusive = results[case][i+1]
            ungrammatical = results[case][i+2]
            if args.unit_type != 'word':
                g_sent = ''.join(''.join(grammatical).split('\s'))
                i_sent = ''.join(''.join(intrusive).split('\s'))
                u_sent = ''.join(''.join(ungrammatical).split('\s'))
            else:
                g_sent = grammatical
                i_sent = intrusive
                u_sent = ungrammatical
            if is_more_probable(grammatical, intrusive):
                sentences['gi_g'][case].append((g_sent, i_sent, u_sent))
            else:
                sentences['gi_i'][case].append((g_sent, i_sent, u_sent))
            if is_more_probable(grammatical, ungrammatical):
                sentences['gu_g'][case].append((g_sent, i_sent, u_sent))
            else:
                sentences['gu_u'][case].append((g_sent, i_sent, u_sent))
            if is_more_probable(intrusive, ungrammatical):
                sentences['iu_i'][case].append((g_sent, i_sent, u_sent))
            else:
                sentences['iu_u'][case].append((g_sent, i_sent, u_sent))
    return [sentences[x] for x in options]

def display_agrmt_results(name, sents):
    # print case-by-case accuracies
    correct_sents, incorrect_sents = sents
    overall_correct = 0.
    total = 0.
    strings = {}
    case_accs = {}
    for case in correct_sents.keys():
        if args.mode != 'overall':
            string = ""
            if len(correct_sents[case]) > 0:
                for i in range(len(correct_sents[case][0][0])):
                    if correct_sents[case][0][0][i][0] == correct_sents[case][0][1][i][0]:
                        string += correct_sents[case][0][0][i][0] + " "
                    else: string += correct_sents[case][0][0][i][0]+"/*"+correct_sents[case][0][1][i][0]+" "
            else:
                for i in range(len(incorrect_sents[case][0][0])):
                    if incorrect_sents[case][0][0][i][0] == incorrect_sents[case][0][1][i][0]:
                        string += incorrect_sents[case][0][0][i][0] + " "
                    else: string += incorrect_sents[case][0][0][i][0]+"/*"+incorrect_sents[case][0][1][i][0]+" "
            strings[case] = string[:-1]
            case_accs[case] = float(len(correct_sents[case]))/(len(correct_sents[case])+len(incorrect_sents[case]))
        overall_correct += len(correct_sents[case])
        total += len(correct_sents[case]) + len(incorrect_sents[case])
    # print case-by-case-accuracies
    if args.mode != 'overall':
        case_out = open(directory+"/case_accs.txt", 'a')
        case_out.write("\n##########\n" + name + "\n##########\n"+"Overall acc: "+str(float(overall_correct)/total)+"\n")
        for (case,score) in sorted(case_accs.items(), key=operator.itemgetter(1)):
            case_out.write(str(case)+":\n")
            case_out.write(strings[case]+": "+str(round(score,4))+"\n")
        case_out.write("\n")
        case_out.close()
    # print individual scores
    if args.mode == 'full':
        fout = open(directory+"/individual_accs.txt", 'a')
        fout.write("\n##########\n" + name + "\n##########\n\n")
        fout.write("Examples that the LM predicts incorrectly:\n")
        for case in incorrect_sents.keys():
            count = 0
            fout.write(case+":\n")
            for good, bad in incorrect_sents[case]:
                if count < 5:
                    fout.write("Grammatical:"+str(round(sum([x[1] for x in good]),2))+"\n")
                    fout.write('\t'.join([x[0] for x in good])+"\n")
                    fout.write('\t'.join([str(round(x[1],2)) for x in good])+"\n")
                    fout.write("Ungrammatical:"+str(round(sum([x[1] for x in bad]),2))+"\n")
                    fout.write('\t'.join([x[0] for x in bad])+"\n")
                    fout.write('\t'.join([str(round(x[1],2)) for x in bad])+"\n")
                    count += 1
                else:
                    break
        fout.write("\n")
        fout.close()
    return float(overall_correct)/total

def display_npi_results(name, sents):
    gi_grammatical_sents, gi_intrusive_sents, iu_intrusive_sents, iu_ungrammatical_sents, gu_grammatical_sents, gu_ungrammatical_sents = sents
    overall_gi = 0.
    overall_iu = 0.
    overall_gu= 0.
    total_gi = 0.
    total_iu = 0.
    total_gu = 0.
    strings = {}
    gi_case_accs = {}
    iu_case_accs = {}
    gu_case_accs = {}
    for case in gi_grammatical_sents.keys():
        gi_case_accs[case] = {}
        iu_case_accs[case] = {}
        gu_case_accs[case] = {}
        if args.mode != "overall":
            if len(gi_grammatical_sents[case]) > 0:
                string = ' '.join([x[0] for x in gi_grammatical_sents[case][0][0]]) + " vs. " + ' '.join([x[0] for x in gi_grammatical_sents[case][0][1]]) + " vs. " + ' '.join([x[0] for x in gi_grammatical_sents[case][0][2]])
            else:
                string = ' '.join([x[0] for x in gi_intrusive_sents[case][0][0]]) + " vs. " + ' '.join([x[0] for x in gi_intrusive_sents[case][0][1]]) + " vs. " + ' '.join([x[0] for x in gi_intrusive_sents[case][0][2]])
            strings[case] = string
            gi_case_accs[case] = float(len(gi_grammatical_sents[case]))/(len(gi_grammatical_sents[case])+len(gi_intrusive_sents[case]))
            iu_case_accs[case] = float(len(iu_intrusive_sents[case]))/(len(iu_intrusive_sents[case])+len(iu_ungrammatical_sents[case]))
            gu_case_accs[case] = float(len(gu_grammatical_sents[case]))/(len(gu_grammatical_sents[case])+len(gu_ungrammatical_sents[case]))
        overall_gi += len(gi_grammatical_sents[case])
        overall_iu += len(iu_intrusive_sents[case])
        overall_gu += len(gu_grammatical_sents[case])
        total_gi += len(gi_grammatical_sents[case]) + len(gi_intrusive_sents[case])
        total_iu += len(iu_intrusive_sents[case]) + len(iu_ungrammatical_sents[case])
        total_gu += len(gu_grammatical_sents[case]) + len(gu_ungrammatical_sents[case])

    # print case-by-case accuracies
    if args.mode != "overall":
        case_out = open(directory+"/case_accs.txt", 'a')
        # print overall accuracies
        case_out.write("\n##########\n" + name + "\n##########\n" + "Overall acc:\n")
        case_out.write("OVERALL P(GRAMMATICAL) > P(INTRUSIVE): "+str(float(overall_gi)/total_gi)+"\n")
        case_out.write("OVERALL P(INTRUSIVE) > P(UNGRAMMATICAL): "+str(float(overall_iu)/total_iu)+"\n")
        case_out.write("OVERALL P(GRAMMATICAL) > P(UNGRAMMATICAL): "+str(float(overall_gu)/total_gu)+"\n")
        for (case,score) in sorted(gi_case_accs.items(), key=operator.itemgetter(1)):
            case_out.write(str(case)+":\n")
            case_out.write(strings[case]+":\n")
            case_out.write("Grammatical > intrusive: "+str(round(score,4))+"\n")
            case_out.write("Grammatical > ungrammatical: "+str(gu_case_accs[case])+"\n")
            case_out.write("Intrusive > ungrammatical: "+str(iu_case_accs[case])+"\n")
        case_out.write("\n")
        case_out.close()
    
    # print individual scores
    if args.mode == "full":
        fout = open(directory+"/individual_accs.txt", 'a')
        fout.write("\n##########\n" + name + "\n##########\n\n")
        fout.write("Examples where the LM prefers the intrusive licensor over the grammatical case:\n")
        for case in gi_intrusive_sents.keys():
            count = 0
            fout.write(case+":\n")
            for c,i,u in gi_intrusive_sents[case]:
                if count < 5:
                    fout.write("grammatical:"+str(round(sum([x[1] for x in c]),2))+"\n")
                    fout.write('\t'.join([x[0] for x in c])+"\n")
                    fout.write('\t'.join([str(round(x[1],2)) for x in c])+"\n")
                    fout.write("intrusive:"+str(round(sum([x[1] for x in i]),2))+"\n")
                    fout.write('\t'.join([x[0] for x in i])+"\n")
                    fout.write('\t'.join([str(round(x[1],2)) for x in i])+"\n")
                    count += 1
                else:
                    break
        fout.write("\nExamples where the LM prefers the ungrammatical case over the intrusive licensor:\n")
        for case in iu_ungrammatical_sents.keys():
            count = 0
            fout.write(case+":\n")
            for c,i,u in iu_ungrammatical_sents[case]:
                if count < 5:
                    fout.write("intrusive:"+str(round(sum([x[1] for x in i]),2))+"\n")
                    fout.write('\t'.join([x[0] for x in i])+"\n")
                    fout.write('\t'.join([str(round(x[1],2)) for x in i])+"\n")
                    fout.write("ungrammatical:"+str(round(sum([x[1] for x in u]),2))+"\n")
                    fout.write('\t'.join([x[0] for x in u])+"\n")
                    fout.write('\t'.join([str(round(x[1],2)) for x in u])+"\n")
                    count += 1
                else:
                    break
        fout.write("\nExamples where the LM prefers the ungrammatical case over the grammatical case:\n")
        for case in gu_ungrammatical_sents.keys():
            count = 0
            fout.write(case+":\n")
            for c,i,u in gu_ungrammatical_sents[case]:
                if count < 5:
                    fout.write("grammatical:"+str(round(sum([x[1] for x in c]),2))+"\n")
                    fout.write('\t'.join([x[0] for x in c])+"\n")
                    fout.write('\t'.join([str(round(x[1],2)) for x in c])+"\n")
                    fout.write("ungrammatical:"+str(round(sum([x[1] for x in u]),2))+"\n")
                    fout.write('\t'.join([x[0] for x in u])+"\n")
                    fout.write('\t'.join([str(round(x[1],2)) for x in u])+"\n")
                    count += 1
                else:
                    break
        fout.write("\n")
        fout.close()
    return float(overall_gi)/total_gi, float(overall_iu)/total_iu, float(overall_gu)/total_gu


# save overall results to file
directory=args.out_dir+"/"+args.model_type+"/"+args.analysis
os.system("mkdir -p " + directory)
with open(directory+"/overall_accs.txt", 'w') as f:
    for name in joined_results.keys():
        if "npi" in name:
            #f.write("############\n"+name+" - NPI:\n############\n")
            sents = analyze_npi_results(joined_results[name])
            overall_gi, overall_iu, overall_gu = display_npi_results(name, sents)
            f.write(name+"(grammatical vs. intrusive): "+str(overall_gi)+"\n")
            f.write(name+"(intrusive vs. ungrammatical): "+str(overall_iu)+"\n")
            f.write(name+"(grammatical vs. ungrammatical): "+str(overall_gu)+"\n")
        else:
            #f.write("############\n"+name+" - SUBJECT/VERB:\n############\n")
            sents = analyze_agrmt_results(joined_results[name])
            overall = display_agrmt_results(name, sents)
            f.write(name+": "+str(overall)+"\n")
