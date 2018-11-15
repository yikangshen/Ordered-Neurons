import logging
import os
logging.basicConfig(level=logging.INFO)

class TestWriter():
    def __init__(self, template_dir, sent_file):
        self.name_lengths = {}
        self.key_lengths = {}
        self.template_dir = template_dir
        self.out_file = os.path.join(self.template_dir, sent_file)
        

    def write_tests(self, all_sents, unit_type):
        logging.info("Writing tests...")
        with open(self.out_file, 'w') as f:
            name_length = 0
            key_length = 0
            for name in all_sents.keys():
                if "npi" in name:
                    multiplier=3
                else: multiplier=2
                self.key_lengths[name] = {}
                for key in all_sents[name].keys():
                    key_length += multiplier * len(all_sents[name][key])
                    self.key_lengths[name][key] = key_length
                    name_length += multiplier * len(all_sents[name][key])
                    for sent in all_sents[name][key]:
                        for i in range(len(sent)):
                            if unit_type != 'word':
                                chars = [x if x != ' ' else '/s' for x in sent[i]+' ']
                                f.write(' '.join(chars)+' . /s <eos>\n')
                            else:
                                f.write(sent[i] + " . <eos>\n")
                    self.name_lengths[name] = name_length
