import os, sys
import logging

sys.path.append(os.path.join(os.path.dirname(__name__), ".."))

from lib.nlp_utils import *

from CORE import CORE
from FTD import FTD

from lib.nlp_utils import *

from dotenv import load_dotenv
load_dotenv()

from typing import Union

from argparse import ArgumentParser


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def main(args):

        mapping = args.mapping
        chunk = args.chunk
        load = args.load
        model_name = args.model_name

        if load == True:

                folder = 'zero-shot_data'
                os.makedirs(folder, exist_ok=True)
                
        #ensemble datasets
        core = CORE(load=load)
        core.ensemble_dataset()
        ftd = FTD(load=load)
        ftd.ensemble_dataset()
       
        if mapping:
                core.map_to_our_labels(mapping=mapping)
                ftd.map_to_our_labels(mapping=mapping)
                logger.info("Finished mapping ...")
        
        core.basic_preprocessing()
        ftd.basic_preprocessing()
        logger.info("Finished deduplication and cleaning ...")

        if chunk:
                chunker = TextChunker(model_name = model_name)
                core.df = chunker.chunker_encoder_based(core.df, col = "text")
                ftd.df = chunker.chunker_encoder_based(ftd.df, col = "text")
                logger.info("Chunking completed ...")

        save_folder = 'zero-shot_data_preprocessed'
        os.makedirs(save_folder, exist_ok=True)

        core.df.to_json(os.path.join(save_folder, "CORE.json"), orient='split', compression='infer')
        ftd.df.to_json(os.path.join(save_folder, "FTD.json"), orient='split', compression='infer')
        

        # print(core.df.head(2), len(core.df), 
        #       ftd.df.head(2), len(ftd.df))
        

if __name__ == '__main__':
        
        parser = ArgumentParser(description='This script loads data for zero-shot experiments from sources and prepares it for classification')
        
        parser.add_argument('--model_name', 
                                type=str, 
                                choices=["FacebookAI/xlm-roberta-base"],
                                default="FacebookAI/xlm-roberta-base",
                                help='Model to use for tokenization')
        parser.add_argument('--load', 
                                type=bool, 
                                help='Load CORE, FTD, UDM from source repositories', 
                                default=False)
        parser.add_argument('--mapping', 
                                type=str,
                                choices = ["cl_map_v1", "cl_map_v2", "cl_map_v3"],
                                help="""Label mapping to map labels from zero-shot datasets (CORE, FTD, UDM) to your dataset's labels \
                                        (for an example of ActDisease maps, check mapping_utils.py). \
                                        Leave as None if no mapping is needed""", 
                                default=None)
        parser.add_argument('--chunk', 
                                type=bool, 
                                help='Chunking of long texts (over 512 tokens)', 
                                default=False)


        args = parser.parse_args()
        main(args)