import torch
from transformers import (AutoTokenizer, 
                          PreTrainedTokenizerFast,
                          AutoModelForSequenceClassification,
                          Trainer,
                          TrainingArguments)
import datasets as ds
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from dataclasses import dataclass
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from typing import List, Optional

from dotenv import load_dotenv
from experimental_dataset import ExperimentalDataset
import os
load_dotenv()

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import yaml
import wandb
wandb.login()

from argparse import ArgumentParser

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

torch.seed(1234)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_config(file, model_name):
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    return config[model_name]

def main(args):

    args.data_path
    model_name = args.model_path
    config = load_config(args.config, os.path.basename(model_name))

    labels = ExperimentalDataset.labels
    
    id2label = dict(enumerate(labels))
    label2id = dict((label, idx) for idx, label in enumerate(labels))

    model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels = len(labels), 
    id2label=id2label, 
    label2id=label2id
    )

    model.to(device)

    training_args = TrainingArguments(
        output_dir=args.output_dir_path,
        **config
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ExperimentalDataset.dataset["train"],
        eval_dataset=ExperimentalDataset.dataset["validation"],
        tokenizer=ExperimentalDataset.tokenizer,
        data_collator=ExperimentalDataset.data_collator
    )

    logger.info(f"Starting training...")

    trainer.train()

    logger.info(f"Saving trainer...")
    trainer.save_model(os.path.join(args.output_dir_path))


if __name__ == '__main__':

    torch.cuda.empty_cache()

    parser = ArgumentParser(description='Training of a genre classification encoder')
    
    parser.add_argument('--model_name', 
                        type=str,
                        choices=["FacebookAI/xlm-roberta-base", "dbmdz/bert-base-historic-multilingual-cased"],
                        default="dbmdz/bert-base-historic-multilingual-cased")
    parser.add_argument('--data_path', 
                        type=str,
                        help="Data folder with train and test .csv file, each containing 'text' and 'label' columns",
                        default="./data")
    parser.add_argument('--output_dir_path', 
                        type=str,
                        help="Data folder to save the resulting models to",
                        default="./models")
    parser.add_argument('--config', 
                        type=str, 
                        help='Path to config', 
                        default='config.yaml')
    
    args = parser.parse_args()

    os.environ["WANDB_PROJECT"]=os.path.basename(args.model_name)

    main(args)


    
        





