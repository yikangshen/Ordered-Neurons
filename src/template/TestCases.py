class TestCase():
    def __init__(self):
        self.agrmt_cases = ['obj_rel_across_anim',
                            'obj_rel_within_anim',
                            'obj_rel_across_inanim',
                            'obj_rel_within_inanim',
                            'subj_rel',
                            'prep_anim',
                            'prep_inanim',
                            'obj_rel_no_comp_across_anim',
                            'obj_rel_no_comp_within_anim',
                            'obj_rel_no_comp_across_inanim',
                            'obj_rel_no_comp_within_inanim',
                            'simple_agrmt',
                            'sent_comp',
                            'vp_coord',
                            'long_vp_coord',
                            'reflexives_across',
                            'simple_reflexives',
                            'reflexive_sent_comp']
        
        self.npi_cases = ['npi_across_anim',
                          'npi_across_inanim',
                          'simple_npi_anim',
                          'simple_npi_inanim']

        self.all_cases = self.agrmt_cases+self.npi_cases
