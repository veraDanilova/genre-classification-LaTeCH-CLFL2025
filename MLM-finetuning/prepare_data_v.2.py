import sys, os, re

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.nlp_utils import *
from lib.utils import get_file_structures
from lib.mapping_utils import *

from tqdm import tqdm
import random
import tempfile
import shutil

import logging
import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import deque

from sklearn.model_selection import train_test_split

from multiprocessing import Pool, cpu_count

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


ACTDISEASE_PATH = os.getenv('TEXT_DATABASE')

BASE_DATA_PATH = Path(ACTDISEASE_PATH)
OUTPUT_DIR = Path("./prepared_data_final")

CUSTOM_TEMP_DIR = Path(os.getenv("TEMP_PATH"))

MODEL_NAME = "bert-base-multilingual-cased" 
MAX_CHUNK_LENGTH = 510

BATCH_SIZE = 10000

TEST_SIZE = 0.1
DEV_SIZE = 0.1

def clean_text(text: str) -> str:
    
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n+', ' ', text).strip()
    return text


def create_chunks(initial_sentences: list[str], 
                  text_tokenizer, 
                  max_chunk_length,
                  text_splitter: RecursiveCharacterTextSplitter) -> list[str]:
    """
    Groups sentences into chunks and chunks sentences longer than the threshold.
    """

    chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0
    sentences_queue = deque(initial_sentences)

    while sentences_queue:
        sentence = sentences_queue.popleft().strip()
        if not sentence:
            continue

        sentence_tokens = len(text_tokenizer.encode(sentence, add_special_tokens=False))

        if sentence_tokens > max_chunk_length:

            # print(f"Sentence with {sentence_tokens} tokens (limit is {MAX_CHUNK_LENGTH}).")
            sub_sentences = text_splitter.split_text(sentence)
            
            sentences_queue.extendleft(reversed(sub_sentences)) # return the sentence's chunks back to the original sentence's position in the correct order
            continue

        # sentence does not enter the current chunk
        if current_chunk_tokens + sentence_tokens > max_chunk_length:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
            
            current_chunk_sentences = [sentence]
            current_chunk_tokens = sentence_tokens
        
        # sentence enters the current chunk
        else:
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sentence_tokens
            
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
        
    return chunks

def process_and_write_batch(chunk_batch, temp_paths, config):
    
    if not chunk_batch:
        return

    random.shuffle(chunk_batch)
    
    train_val_chunks, test_chunks = train_test_split(chunk_batch, test_size=config['test_size'], random_state=42)
    train_chunks, dev_chunks = train_test_split(train_val_chunks, test_size=config['dev_size'], random_state=42)

    with temp_paths['train'].open("a", encoding="utf-8") as f:
        f.write("\n".join(train_chunks) + "\n")
    with temp_paths['dev'].open("a", encoding="utf-8") as f:
        f.write("\n".join(dev_chunks) + "\n")
    with temp_paths['test'].open("a", encoding="utf-8") as f:
        f.write("\n".join(test_chunks) + "\n")


# file_list = get_file_structures(ACTDISEASE_PATH)
# print(f"Found {len(file_list)} files in the directory '{os.path.abspath(ACTDISEASE_PATH)}'.\n")

def process_folder_task(args):
    """
    Processes one folder and writes chunks directly to temporary files on disk.
    """
    folder_name, config = args

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    print(f"{folder_name}: Loaded the tokenizer: {MODEL_NAME}")

    # ------------------------------------
    langchain_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_LENGTH,
        chunk_overlap=int(MAX_CHUNK_LENGTH * 0.1), #10% of chunk length overlap
        length_function=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        separators=["\n\n", "\n", ".", "?", "!", ",", " "],
        keep_separator = False
    )
    print(f"{folder_name}: Initialized the RecursiveCharacterTextSplitter ...")
    # ------------------------------------

    language = config['org_df'][config['org_df'].folders == folder_name].language.iloc[0]
    wtp_code = config['lang_map'].get(language)
    # ------------------------------------

    print(f"{folder_name}: Initializing wtpsplit for: '{language}' (code: {wtp_code})")
    sat = SaT("sat-12l", style_or_domain="ud", language=wtp_code)
    print(f"{folder_name}: Initialized sentence splitter ...")
    # ------------------------------------

    # Create a unique temporary dir
    temp_dir = Path(tempfile.mkdtemp(dir=config['custom_temp_dir']))

    temp_paths = {
        'train': temp_dir / "train.txt",
        'dev': temp_dir / "dev.txt",
        'test': temp_dir / "test.txt",
    }

    org_folder_path = config['base_path'] / folder_name
    if not org_folder_path.exists():
        return {'status': 'skip', 'reason': f"Folder {org_folder_path} not found."}

    chunk_batch = []

    try:
        txt_files = list(org_folder_path.rglob("*.txt"))
        for txt_file in txt_files:
            try:
                raw_text = txt_file.read_text(encoding="utf-8")
                cleaned_text = clean_text(raw_text)
                if not cleaned_text: continue
                
                initial_sentences = sat.split(cleaned_text)
                file_chunks = create_chunks(initial_sentences, tokenizer, config['max_chunk_length'], langchain_splitter)
                
                chunk_batch.extend(file_chunks)
                
                
                if len(chunk_batch) >= config['batch_size']:
                    process_and_write_batch(chunk_batch, temp_paths, config)
                    chunk_batch = [] # Очищаем буфер

            except Exception as e:
                print(f"Warning: Failed to process file {txt_file}. Error: {e}")

        
        if chunk_batch:
            process_and_write_batch(chunk_batch, temp_paths, config)

        return {
            'status': 'success',
            'temp_dir': str(temp_dir),
        }
    except Exception as e:
        shutil.rmtree(temp_dir)
        return {'status': 'error', 'file': str(org_folder_path), 'error': f"[{type(e).__name__}] {e}"}

def main():

    global organizations
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    train_file = OUTPUT_DIR / "train.txt"
    dev_file = OUTPUT_DIR / "dev.txt"
    test_file = OUTPUT_DIR / "test.txt"

    
    for f in [train_file, dev_file, test_file]: f.write_text("")

    organizations = organizations.explode('folders').dropna(subset="folders", axis=0)
    unique_organization_folders = organizations['folders'].unique()

    config = {
        'model_name': MODEL_NAME,
        'max_chunk_length': MAX_CHUNK_LENGTH,
        'base_path': BASE_DATA_PATH,
        'batch_size': BATCH_SIZE,
        'org_df': organizations,
        'lang_map': wtp_language_codes,
        'test_size': TEST_SIZE,
        'dev_size': DEV_SIZE,
        'custom_temp_dir': str(CUSTOM_TEMP_DIR)
    }

    tasks = [(folder_name, config) for folder_name in unique_organization_folders]
    
    num_processes = max(1, cpu_count() - 1) # one cpu free
    print(f"Processing started in {num_processes} processes...")

    # Process pool
    with Pool(processes=num_processes) as pool:
        # imap_unordered - serve as soon as ready
        results = list(tqdm(pool.imap_unordered(process_folder_task, tasks), total=len(tasks), desc="Total progress"))

    print("\nProcessing complete. Concatenating temporary files...")
    
    # Efficiently concatenate temporary files into final files
    for result in tqdm(results, desc="Writing final files"):

        if result['status'] == 'success':
            temp_dir = Path(result['temp_dir'])
            try:
                
                with train_file.open("ab") as f_out, (temp_dir / "train.txt").open("rb") as f_in:
                    shutil.copyfileobj(f_in, f_out)
                
                with dev_file.open("ab") as f_out, (temp_dir / "dev.txt").open("rb") as f_in:
                    shutil.copyfileobj(f_in, f_out)
                
                with test_file.open("ab") as f_out, (temp_dir / "test.txt").open("rb") as f_in:
                    shutil.copyfileobj(f_in, f_out)
            finally:
                # Clean up temp dir
                shutil.rmtree(temp_dir)

        elif result['status'] == 'error':
            print(f"Error during processing folder for {result.get('file', 'N/A')}: {result['error']}")

    print(f"\nData for MLM finetuning saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
