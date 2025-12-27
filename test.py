import soundfile as sf
import numpy as np
from datasets import load_dataset, Audio
from IPython.display import Audio as IPyAudio



file_path = 'clean/test-00000-of-00009.parquet'

train_ds = load_dataset("MLCommons/peoples_speech", data_files=file_path, split='train') # downloading one specific file
print(train_ds.column_names)


# x = np.zeros(16000)
# sf.write('test.wav', x, 16000)
# data, sr = sf.read("test.wav")
# print(sr)