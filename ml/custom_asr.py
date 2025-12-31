from datasets import load_dataset, Audio
import os
from dotenv import load_dotenv
from pathlib import Path
import io
import soundfile as sf

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
MODEL = '' # bert

one_file = 'main/train/0000.parquet'
cache_path = f'{Path.cwd()}/cache' 

dataset = load_dataset("keithito/lj_speech", 
                       revision='refs/convert/parquet',
                       data_files=one_file,
                       split='train',
                       cache_dir=cache_path
                       )
dataset = dataset.cast_column("audio", Audio(decode=False)) # disable automatic decoding

# 

# verifying steps
sample = next(iter(dataset)) # one file
audio = sample['audio'] # type: ignore
real_text = sample['text'] # type: ignore
print(real_text)

with open("test_output.wav", 'wb') as f:
    f.write(audio['bytes'])
    
# lj speech dataset
# https://huggingface.co/datasets/keithito/lj_speech/tree/refs%2Fconvert%2Fparquet

