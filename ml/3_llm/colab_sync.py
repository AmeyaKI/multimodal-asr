import shutil, os
from pathlib import Path

path = '/Users/ameyakiwalkar/Google Drive/My Drive/Colab Notebooks/data/asr-v1'
print(list(Path(path).iterdir()))

# USE THIS TO UPLOAD ASR OUTPUT FILES TO FEED TO LLM