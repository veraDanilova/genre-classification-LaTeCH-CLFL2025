import os, sys, re

from collections import defaultdict
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from ZeroShotDataset import *
from loader import *

def check_nmax(df):
    return len([val for val in df['A1':'A17'] if val == df['max']])

class FTD(ZeroShotDataset):

    def __init__(self, 
                 load=False):
        super().__init__()
        self.dataset_name = "FTD"
        self.load = load
        self.dataset_folder = os.path.join(os.path.join(os.path.dirname(__file__), self.dataset_folder))


        if self.load == True:
            self._load_dataset()

    def _load_dataset(self):

        load_FTD(self.dataset_folder)

    def ensemble_dataset(self):

        print(f'Ensembling the FTD dataset ..')

        self.dataset_folder = os.path.join(self.dataset_folder, self.dataset_name)
        
        # English
        with open(os.path.join(self.dataset_folder,'en.ol'),'r') as f:
            ftd_en = f.readlines()
        with open(os.path.join(self.dataset_folder,'unzipped_en.gold.dat'),'r', encoding='utf-8') as f:
            ftd_en_labels = f.readlines()

        #Russian
        with open(os.path.join(self.dataset_folder, 'ru.ol'),'r') as f:
            ftd_ru = f.readlines()
        ftd_ru_labels = pd.read_csv(os.path.join(self.dataset_folder,'ru.csv'), index_col=0,sep='\t')

        # ftd_ru_labels['npred'] = ftd_ru_labels.apply(check_nmax,axis=1)
        ftd_ru_labels['label'] = ftd_ru_labels.loc[:,'A1':'A17'].idxmax(axis=1)
        
        self.df = pd.DataFrame(
            {
            'text' : ftd_ru+ftd_en,
            'language': ["Russian"]*len(ftd_ru)+['English']*len(ftd_en),
            'label': ftd_ru_labels['label'].tolist()+[re.sub('\n','',label) for label in ftd_en_labels]
            }
        )
        
        print('Done. ')

