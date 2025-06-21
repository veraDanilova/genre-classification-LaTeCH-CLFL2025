import os, sys, re

from loader import *

from collections import defaultdict
import pandas as pd

from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.mapping_utils import *
from lib.nlp_utils import *

class ZeroShotDataset:

    def __init__(self, 
                 dataset_name: str = None, 
                 language: str = None, 
                 dataset_folder: str = 'zero-shot_data'):

        self.dataset_name = dataset_name
        self.language = language
        self.df: Optional[pd.DataFrame] = None
        self.dataset_folder = dataset_folder

    def map_labels(self, row: pd.Series, mapping: defaultdict = None) -> str:

        label = row["original_label"]
        result = list(filter(lambda k: label in maps[mapping][k][self.dataset_name], maps[mapping].keys()))
        output_key = result[0] if result else None
        return output_key

    def map_to_our_labels(self, mapping: defaultdict = None):

        self.df.rename({'label':'original_label'}, axis=1, inplace=True)
        self.df['label'] = self.df.apply(lambda row: self.map_labels(row, mapping), axis=1)
    
    def nans_and_duplicates(self):

        self.df.dropna(inplace=True)
        self.df.drop_duplicates(subset=["text"], inplace=True)

    def map_to_language_families(self):

        self.df['language_family'] = self.df.language.apply(lambda l: 
                                                        next((key 
                                                            for key in language_families.keys() 
                                                            if l in language_families[key]), None))
    def basic_preprocessing(self):

        """Drop nans and duplicates,
        Map language families"""

        self.nans_and_duplicates()
        self.map_to_language_families()

        if "text" in self.df.columns:
            self.df["text"] = clean_sentences(self.df["text"].tolist())
        else:
            raise ValueError("'text' column is missing from the dataframe ... ")