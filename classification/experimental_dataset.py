from transformers import (PreTrainedTokenizerFast,
                          AutoTokenizer,
                          DataCollatorWithPadding)
from datasets import DatasetDict
import pandas as pd
from typing import Optional, List
from sklearn.preprocessing import LabelEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
import datasets as ds

class ExperimentalDataset:

    def __init__(self,
                 train_df: Optional[pd.DataFrame] = None,
                 do_test_split: bool = True,
                 checkpoint_name: str = "dbmdz/bert-base-historic-multilingual-cased"):

        self.train_df: Optional[pd.DataFrame] = train_df.copy() if train_df is not None else None
        
        self.checkpoint_name: str = checkpoint_name

        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self.checkpoint_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.dataset: Optional[ds.Dataset] = None
        self.labels: List[str] = []

        self.do_test_split = do_test_split
    
    def text_splitter(self,
                      chunk_size: int = 490,
                      chunk_overlap: int = 20,
                      separators: Optional[List[str]] = None) -> RecursiveCharacterTextSplitter:
        
        if separators is None:
            separators = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda text: len(self.tokenizer.encode(text, add_special_tokens=False)),
            separators=separators,
            keep_separator=False
        )
    
    def prepare_dataset(self):
        
        self.labels = self.train_df.label.unique()
        label = ds.ClassLabel(num_classes=len(self.labels), names=self.labels)
        dataset = ds.Dataset.from_pandas(self.data_df)

        def preprocess(df):
            return self.tokenizer(dataset["text"], truncation=True) 
        
        dataset = dataset.map(preprocess, batched=True)
        dataset = dataset.cast_column("label", label)

        dataset = dataset.train_test_split(test_size=0.2, 
                                                stratify_by_column=self.stratify_column,
                                                seed=0)
        if self.do_test_split:
            
            test_split = dataset['test'].train_test_split(test_size=0.5,
                                                                    stratify_by_column=self.stratify_column,
                                                                    seed=0)

            self.dataset = DatasetDict({
                'train': dataset['train'],
                'test': test_split['test'],
                'validation': test_split['train']})
             
        else:

            self.dataset = DatasetDict({
                'train': dataset['train'],
                'validation': dataset['test']
            })

        splits = list(self.dataset.keys())
        list(map(lambda split: self.dataset[split].set_format("torch", columns=["input_ids", "attention_mask", "label"]), splits))