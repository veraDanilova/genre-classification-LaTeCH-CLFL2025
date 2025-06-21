import os, sys

from collections import defaultdict
import pandas as pd


sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from ZeroShotDataset import *
from loader import *


def expand_rows(row, lang, dataset_name):
    
    labels = row['label'].split()
    text = row['text']
    lang = lang
    expanded_rows = [{'dataset': dataset_name, 'language': lang, 'label': label, 'text': text} for label in labels]

    return expanded_rows

class CORE(ZeroShotDataset):

    def __init__(self, 
                load=False):
        super().__init__()
        self.dataset_name = "CORE"
        self.dataset_folder = os.path.join(os.path.join(os.path.dirname(__file__), self.dataset_folder))

        self.load = load

        if self.load == True:
            self._load_dataset()

    def _load_dataset(self):

        load_CORE(self.dataset_folder)

    def ensemble_dataset(self):

        data_splits = ['train.tsv','dev.tsv']

        data = []

        lang_mapping = {'Fin': 'Finnish',
                        'Swe': 'Swedish',
                        'Fre': 'French'}

        for folder in os.listdir(self.dataset_folder):

            if folder == "CORE":

                language = 'English'

                for split in data_splits:

                    df = pd.read_csv(os.path.join(self.dataset_folder, folder, split), sep='\t',
                    names=['label', 'id', 'text']).dropna(subset="label")
                    df = pd.DataFrame([item for sublist in df.apply(lambda r: expand_rows(r, lang=language, dataset_name="CORE"), axis=1) for item in sublist])
                    data+=[df]

            elif folder != "CORE" and "CORE" in folder:

                lang = folder.split('CORE')[0]

                for split in data_splits:

                    df = pd.read_csv(os.path.join(self.dataset_folder, folder, split), sep='\t', encoding='utf-8',
                                    names=['label', 'text']).dropna(subset="label")
                    
                    df = pd.DataFrame([item for sublist in df.apply(lambda r: expand_rows(r, lang=lang_mapping[lang], dataset_name=folder), axis=1) for item in sublist])
                    data+=[df]
        
        print(f'Ensembling the CORE dataset (train and dev splits)..')
        self.df = pd.concat(data, axis=0)
        print('Done. ')