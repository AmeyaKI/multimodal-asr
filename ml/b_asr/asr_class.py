import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForCTC
import librosa

class ASR():
  DEFAULT_MODEL = 'nvidia/parakeet-ctc-1.1b' # train custom model
  DEFAULT_DEVICE = 'cuda'
  TARGET_SAMPLE_RATE = 16000 # resample to 16,000 hz

  
  def __init__(self, custom_model=None, device=None):
    self.model_name = custom_model or self.DEFAULT_MODEL
    self.device = device or self.DEFAULT_DEVICE

    self.processor = AutoProcessor.from_pretrained(self.model_name)
    self.model = AutoModelForCTC.from_pretrained(self.model_name, dtype='auto').to(self.device) 
    self.model.eval()

  def process_audio(self, audio_path):
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
          

  def transcribe(self, audio_path):
    waveform = self.process_audio(audio_path)

    # calc model inputs
    model_inputs = self.processor(
        waveform,
        sampling_rate = self.TARGET_SAMPLE_RATE,
        return_tensors = 'pt'
        )
    
    model_inputs = model_inputs.to(
        device = self.model.device,
        dtype = self.model.dtype
        )
        
    with torch.no_grad(): # obtain logits
        # Ensure input_features are in bfloat16 to match model's dtype
        model_inputs.input_features = model_inputs.input_features.to(torch.bfloat16)
        logits = self.model(**model_inputs).logits

    # decode and translate logits to predicted text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = self.processor.batch_decode(predicted_ids)
    predicted_text = transcription[0]
    
    return predicted_text