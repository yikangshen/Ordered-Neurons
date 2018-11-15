import random
import pickle
import sys

from template.Terminals import NPITerminals, AgreementTerminals
from template.Templates import NPITemplate, AgreementTemplate
from template.TestCases import TestCase
                          
class MakeAgreementTemplate():
    def __init__(self):
        self.terminals = AgreementTerminals().terminals
        self.rules = AgreementTemplate().rules

    def switch_number(self, wrds, verb=False):
        new_wrds = []
        for wrd in wrds:
            if wrd.split()[0] == "is":
                new_wrds.append(' '.join(['are']+wrd.split()[1:]))
            elif verb:
                if len(wrd.split()) > 1:
                    new_wrds.append(' '.join([wrd.split()[0][:-1]]+wrd.split()[1:]))
                else:
                    new_wrds.append(wrd[:-1])
            elif wrd[-4:] == "self":
                new_wrds.append("themselves")
            else:
                new_wrds.append(wrd+"s")
        return new_wrds

    def get_case_name(self, preterms, match, vary, opt='sing', v_opt='sing'):
        sent = opt+"_"
        for j in range(len(match)):
            for i in range(len(match[j])):
                sent += preterms[match[j][i]]+"_"
        if len(vary) > 0:
            sent += v_opt + "_"
            for j in range(len(vary)):
                sent += preterms[vary[j]]+"_"
        return sent[:-1]
        

    def switch_numbers(self, base_sent, variables, preterms):
        new_sent = base_sent[:]
        for idx in variables:
            new_sent[idx] = self.switch_number(new_sent[idx], preterms[idx][-1] == "V")
        return new_sent
    
    def make_variable_sents(self, preterms, match, vary):
        all_sents = {}
        base_sent = [self.terminals[p] for p in preterms]
        prefixes = ['sing', 'plur']
        for i in range(2):
            s_grammatical = base_sent[:]
            p_grammatical = self.switch_numbers(base_sent, vary, preterms)

            s_ungrammatical = self.switch_numbers(s_grammatical, match[1], preterms)
            p_ungrammatical = self.switch_numbers(p_grammatical, match[1], preterms)
            
            if i == 1:
                s_ungrammatical = self.switch_numbers(s_grammatical, match[0], preterms)
                p_ungrammatical = self.switch_numbers(p_grammatical, match[0], preterms)
                
                s_grammatical = self.switch_numbers(s_grammatical, match[0]+match[1], preterms)
                p_grammatical = self.switch_numbers(p_grammatical, match[0]+match[1], preterms)
            all_sents[self.get_case_name(preterms, match, vary, opt=prefixes[i], v_opt='sing')] = [s_grammatical, s_ungrammatical]
            if len(vary) > 0:
                all_sents[self.get_case_name(preterms, match, vary, opt=prefixes[i], v_opt='plur')] = [p_grammatical, p_ungrammatical]
            
        return all_sents

class MakeNPITemplate():
    def __init__(self):
        self.terminals = NPITerminals().terminals
        self.rules = NPITemplate().rules

    def switch_tense(self, preterms):
        new_preterms = preterms[:]
        new_preterms[new_preterms.index('PASTAUX')] = 'FUTAUX'
        if 'APMV' in preterms:
            new_preterms[new_preterms.index('APMV')] = 'AFMV'
        else:
            new_preterms[new_preterms.index('IPMV')] = 'IFMV'
        return new_preterms

    def switch_dets(self, preterms, opt=''):
        new_preterms = preterms[:]
        if opt == 'intrusive':
            new_preterms[new_preterms.index('NO')] = 'SD'
        elif opt == 'ungram':
            new_preterms[new_preterms.index('NO')] = 'MOST'
        return new_preterms

    def make_variable_sents(self, preterms, simple=False):
        all_sents = {}
        prefixes = ['past', 'future']
        p_grammatical = [self.terminals[p] for p in preterms]
        f_grammatical = [self.terminals[p] for p in self.switch_tense(preterms)]

        p_intrusive = [self.terminals[p] for p in self.switch_dets(preterms, opt='intrusive' if simple else '')]
        f_intrusive = [self.terminals[p] for p in self.switch_tense(self.switch_dets(preterms, opt='intrusive' if simple else ''))]

        p_ungrammatical = [self.terminals[p] for p in self.switch_dets(preterms, opt='ungram')]
        f_ungrammatical = [self.terminals[p] for p in self.switch_tense(self.switch_dets(preterms, opt='ungram'))]

        all_sents['past'] = [p_grammatical, p_intrusive, p_ungrammatical]
        all_sents['future'] = [f_grammatical, f_intrusive, f_ungrammatical]

        return all_sents


class MakeTestCase():
    def __init__(self, template, test_case):
        self.template = template
        self.test_case = test_case
        self.sent_templates = self.get_rules()
        
    def get_rules(self):
        sent_templates = {}
        preterminals, templates = self.template.rules[self.test_case]
        if templates is not None:
            sents = self.template.make_variable_sents(preterminals, templates['match'], templates['vary'])
            for k in sents.keys():
                if k not in sent_templates:
                    sent_templates[k] = []
                gram = list(self.expand_sent(sents[k][0]))
                ungram = list(self.expand_sent(sents[k][1]))
                for i in range(len(gram)):
                    sent_templates[k].append((gram[i],ungram[i]))
        else:
            sents = self.template.make_variable_sents(preterminals, simple=self.test_case.startswith('simple'))
            for k in sents.keys():
                if k not in sent_templates:
                    sent_templates[k] = []
                gram = list(self.expand_sent(sents[k][0]))
                intrusive = list(self.expand_sent(sents[k][1], partial= "", switch_ds=not self.test_case.startswith('simple')))
                ungram = list(self.expand_sent(sents[k][2]))
                for i in range(len(gram)):
                    sent_templates[k].append((gram[i],intrusive[i],ungram[i]))
        return sent_templates
    
    def expand_sent(self, sent, partial="", switch_ds=False):
        if len(sent) == 1:
            for wrd in sent[0]:
                if switch_ds:
                    sp = partial.split(" ")
                    no = sp[0]
                    the = sp[3]
                    new_partial_one = ' '.join([x for x in partial.split()[1:3]])
                    new_partial_two = ' '.join([x for x in partial.split()[4:]])
                    yield ' '.join([the, new_partial_one, no, new_partial_two, wrd])

                
                # We want to avoid repeating words/phrases multiple times in the sentences
                # but some words are allowed to repeat, such as determiners or complementizers
                # We also need to check that the phrase isn't repeated save for number
                # e.g. 'the man who the guards like likes pizza'
                # not all sentences with repeating phrases are bad, but many seem implausible
                # so we do not generate them!
                elif wrd not in partial and wrd not in self.template.terminals['D'] and wrd not in self.template.terminals['C'] and not (wrd.split(" ")[0]+"s "+' '.join(wrd.split(" ")[1:]) in partial or wrd.split(" ")[0][:-1]+" "+' '.join(wrd.split(" ")[1:]) in partial) and not ((wrd.startswith('is') and 'are '+wrd[3:] in partial) or (wrd.startswith('are') and 'is '+wrd[4:] in partial)):
                    yield partial + wrd
                else:
                    yield "None"
        else:
            for wrd in sent[0]:
                for x in self.expand_sent(sent=sent[1:], partial=partial + wrd + " ", switch_ds=switch_ds):
                    if x != "None":
                        yield x


def main():
    agrmt_template = MakeAgreementTemplate()
    npi_template = MakeNPITemplate()

    testcase = TestCase()
    
    agrmt_test_cases = testcase.agrmt_cases
    npi_test_cases = testcase.npi_cases
    
    if len(sys.argv) != 2:
        print("Usage: python make_templates.py [template_dir]")
        sys.exit(1)
        
    out_dir = sys.argv[1]
        
    for case in agrmt_test_cases:
        print("case:",case)
        sents = MakeTestCase(agrmt_template, case)
        with open(out_dir+"/"+case+".pickle", 'wb') as f:
            pickle.dump(sents.sent_templates, f)
    for case in npi_test_cases:
        print("case:",case)
        sents = MakeTestCase(npi_template, case)
        with open(out_dir+"/"+case+".pickle", 'wb') as f:
            pickle.dump(sents.sent_templates, f)
            
            
if __name__ == "__main__":
    main()
