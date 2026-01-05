from transformers import pipeline, AutoProcessor, AutoModel
import torch
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

class TTS():
    MODEL = 'suno/bark-small'
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    def __init__(self, model_param=None):
        self.model_name = model_param or self.MODEL
        

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, 
                                          dtype=torch.float32)
        
    def generate_audio(self, prompt):
        inputs = self.processor(
            text=[prompt],
            voice_preset='v2/en_speaker_6',
            return_tensors='pt'
        )

        speech_values = self.model.generate(**inputs, do_sample=True)
        audio_data = speech_values.cpu().numpy().squeeze()
        sampling_rate = self.model.generation_config.sample_rate

        def save_or_play(param):
            if param == 'play':
                sd.play(audio_data, samplerate=sampling_rate)
            elif param == 'save':
                wav_write("output4.wav", rate=sampling_rate, data=audio_data)
            else:
                raise ValueError

        return save_or_play
