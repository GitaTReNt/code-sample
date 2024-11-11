import numpy as np

class DominGenerator:
    '''
    Domin Generator
    '''
    def __init__(self, len_list):
        self.l_list = len_list

    def get_domain_labels(self):
        domain_labels = []
        for i, l in enumerate(self.l_list):
            labels = np.zeros((l, len(self.l_list)), dtype=int)
            labels[:, i] = 1
            domain_labels.append(labels)
        return np.concatenate(domain_labels)
