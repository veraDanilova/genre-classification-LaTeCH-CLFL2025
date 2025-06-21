import os, sys, re
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from typing import Optional, List

from transformers import AutoTokenizer
# import tiktoken
from datasets import Dataset

from wtpsplit import SaT
import pandas as pd
from whitespace_correction import WhitespaceCorrector

from lib.utils import *

from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

from pandarallel import pandarallel

# Initialize pandarallel with the number of workers you want to use
pandarallel.initialize(progress_bar=True)

from huggingface_hub import login
login(token=os.getenv("HUGGINGFACE_TOKEN"))

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# sentence segmentation
def sentence_segmentation(texts: str, lang: str)  -> List[List]:
    
    logger.info(f"Loading wtpsplit model for the language {lang} ...")

    sat = SaT("sat-12l", style_or_domain="ud", language=lang)
    sat.half().to("cuda")

    texts = texts.split('\n\n')

    progress_counter = percentage_coroutine(len(texts))

    def segment_entries(entry_strings):
        
        return [sat.split(entry_string) for entry_string in entry_strings]
    
    sentence_lists = trace_progress(segment_entries(texts), progress = progress_counter)

    logger.info("Finished sentences segmentation")

    return sentence_lists



# whitespace correction
def whitespace_correction(df: Optional[pd.DataFrame], col: str, wsc_model: str = "eo_medium_byte") -> Optional[pd.DataFrame]:

    logger.info(f"Loading whitespace correction model ...")

    cor = WhitespaceCorrector.from_pretrained(model=wsc_model)
    cor.to("cuda:0")
    cor.set_precision("fp32")

    progress_counter = percentage_coroutine(len(df))

    def  correct_entry(sentences):
        return [cor.correct_text(sentence) for sentence in sentences]
    
    df['whitespace_normalized'] = df[col].parallel_apply(trace_progress(correct_entry, progress = progress_counter))

    logger.info("Finished whitespace correction. ")

    return df


class TextChunker:

    def __init__(self, model_name: str = "FacebookAI/xlm-roberta-base"):

        logger.info(f"Initilizing tokenizer for the model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


    # calculate tokens encoders
    def calculate_tokens_encoders(self, df: Optional[pd.DataFrame], col: str)  -> Optional[pd.DataFrame]:

        def span_len(text):
            return len(self.tokenizer.encode(text))
        
        df['token_length'] = df[col].parallel_apply(span_len)

        return df
    
    # calculate tokens for a decoder
    def calculate_tokens_decoders(self, dataset: Optional[Dataset], col: str = None) -> Dataset:

        def measure_token_length(example):
            
            tokens = self.tokenizer(example[col], truncation=False, add_special_tokens=True)
            example["input_ids"] = tokens["input_ids"]
            example["token_length"] = len(tokens["input_ids"])
            return example
        
        return dataset.map(measure_token_length)

    # chunking 
    def chunker_langchain(self, df: Optional[pd.DataFrame], col: str, language: str = "English", chunk_size: int = 512, chunk_overlap: int = 128) -> pd.DataFrame:

        logger.info(f"Initilizating text splitter")

        # separators = RecursiveCharacterTextSplitter.get_separators_for_language(language)

        def span_len(text):
            return len(self.tokenizer.encode(text))

        text_splitter = RecursiveCharacterTextSplitter(
            keep_separator=True,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=span_len,
                separators=["\n\n", ".", "!", "?", ";", ":", ",", " "]
            )
            
        def chunk_text(df):
                
            text = df[col]
            ntokens = df.token_length
            max_length = chunk_size

            if ntokens // 2 >= (max_length - 2):
                return text_splitter.split_text(text)
            else:
                return [df[col]]    
                
        df['chunks'] = df.parallel_apply(chunk_text, axis=1)

        logger.info(f"Finalized text splitter")

        return df
    
    def chunker_encoder_based(self, df: Optional[pd.DataFrame], col: str) -> Optional[pd.DataFrame]:

        df = self.calculate_tokens_encoders(df, col)
        df = self.chunker_langchain(df, col)

        return df


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def clean_sentences(sents):
        
    clean_sents = []
    for sent in sents:
        sent = re.sub(r'<a_href=\S+', '', sent)
        sent = re.sub(r'<(/)?\w+>', '', sent)
        sent = re.sub("[#*]+", "", sent)
        sent = re.sub(r"\[\s?\.+\s?\]", "", sent)
        pattern = r'/\w+/\s?|\(\s*ISBN.*?\)s?'
        sent = re.sub(pattern, '', sent)
        sent = remove_emoji(sent)
        clean_sents.append(sent)
        
    return clean_sents