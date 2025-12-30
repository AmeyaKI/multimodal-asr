from datasets import load_dataset
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

one_file = 'main/train/0000.parquet'
cache_path = f'{os.getcwd()}/cache' 

dataset = load_dataset("keithito/lj_speech", 
                       revision='refs/convert/parquet',
                       data_files=one_file,
                       split='train',
                       cache_dir=cache_path
                       )

print(dataset)
print(dataset['audio'])
print(dataset['text'])
print(dataset['normalized_text'])





#  lj speech
# https://huggingface.co/datasets/keithito/lj_speech/tree/refs%2Fconvert%2Fparquet
