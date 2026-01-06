import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForCTC
import librosa
from pathlib import Path

class ASR():
  DEFAULT_MODEL = 'nvidia/parakeet-ctc-0.6b' # alt: openai/whisper-tiny.en
  TARGET_SAMPLE_RATE = 16000 # hz
  CACHE_DIR = Path.cwd()
  
  def __init__(self, custom_model=None):
    self.model_name = custom_model or self.DEFAULT_MODEL
    self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    self.processor = AutoProcessor.from_pretrained(self.model_name,
                                                  cache_dir=self.CACHE_DIR
                                                  )
    self.model = AutoModelForCTC.from_pretrained(self.model_name, 
                                                dtype=torch.float32,
                                                cache_dir=self.CACHE_DIR
                                                ).to(self.device) 
    self.model.eval()


  def process_arr(self, audio_arr):
    # process audio arr
    return self.transcribe(audio_arr)

  def transcribe(self, waveform):
    model_inputs = self.processor(
        waveform,
        sampling_rate = self.TARGET_SAMPLE_RATE,
        return_tensors = 'pt'
        )
    
    model_inputs = model_inputs.to(device=self.model.device) 
        
    with torch.no_grad():
      logits = self.model(**model_inputs).logits

    # decode and translate logits to predicted text
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_text = self.processor.decode(predicted_ids[0],
                                                skip_special_tokens=True)
    
    return predicted_text


  def process_wav(self, audio_path):
    # process .wav file
    if audio_path is None:
      raise ValueError

    # load audio
    waveform, sample_rate = librosa.load(
        audio_path,
        sr=None,
        mono=True)
    
    # resample if sample_rate is incorrect
    if sample_rate != self.TARGET_SAMPLE_RATE:
        waveform = librosa.resample(
            waveform,
            orig_sr = sample_rate,
            target_sr = self.TARGET_SAMPLE_RATE)
    
    # normalizing audio amp
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform /= max_val
    return waveform
          