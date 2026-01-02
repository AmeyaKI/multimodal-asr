import sounddevice as sd
import numpy as np
from queue import Queue
import torch
import scipy.io.wavfile as sci_wav
from silero_vad import load_silero_vad
from pathlib import Path

vad_model = load_silero_vad()
class MicVAD():
    SAMPLE_RATE = 16000
    FRAME_SIZE = 512 
    VAD_THRESHOLD = 0.5
    OUTPUT_PATH = f'{Path.home()}/Google Drive/My Drive/Colab Notebooks/assistant/vad_recordings' # Google Drive Desktop
    
    def __init__(self, vad_model):
        self.audio_queue = Queue()
        self.vad_model = vad_model
        
        self.frame_duration = self.FRAME_SIZE / self.SAMPLE_RATE

        self.recorded_frames = []
        
    
    def start_microphone(self, max_silence=3.0, output_path=OUTPUT_PATH):
        
        max_silent_frames = int(max_silence / self.frame_duration)
        print("Recording! Start speaking.")
        
        def audio_callback(indata, frames, time, status):
            frame = indata.copy().squeeze()
            if len(frame) < 160:
                return
            self.audio_queue.put(frame)

        silencio_counter = 0
        with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1, 
                            blocksize=self.FRAME_SIZE, dtype="float32", 
                            callback=audio_callback):
            try:
                while True:
                    frame = self.audio_queue.get()
                    
                    audio_tensor = torch.from_numpy(frame).float() # convert audio to tensor

                    with torch.no_grad(): # calculate probabilities
                        speech_prob = self.vad_model(audio_tensor, self.SAMPLE_RATE).item()

                    if speech_prob > self.VAD_THRESHOLD: # if probability audio tensor is speech > 0.5 --> saved in recorded frames
                        self.recorded_frames.append(frame)
                        silencio_counter = 0
                    else:
                        silencio_counter += 1
                        
                    # stop recording if silence max reached: 3 sec
                    if silencio_counter >= max_silent_frames:
                        print(f"{max_silence} seconds of silence reached. Recording stopping.")
                        break

            except KeyboardInterrupt: # stop recording manually
                print("\nStopped recording.")
                
        return self.save_recordings(output_path)

    # save recordings to .wav file
    def save_recordings(self, output_path='../5_outputs'):
        file_name = ''
        if self.recorded_frames:
            # convert recorded frames to audio array
            audio = np.concatenate(self.recorded_frames)
            audio_arr = (audio * 32767).astype(np.int16)
            
            file_num = str(len(list(Path(output_path).iterdir()))).zfill(5)
            file_name = f"{output_path}/file{file_num}.wav"
            
            sci_wav.write(file_name, self.SAMPLE_RATE, audio_arr)
            print(f"Saved speech audio to {output_path}")
            
        else:
            print("No speech detected. No files saved")
            
        self.recorded_frames = [] # RESOLVE LATER: delete all frames in instance of MicVAD ?
        return file_name
    

def stitcher(): # integrate into uniform main.py file later
    micvad = MicVAD(vad_model)
    file_name = micvad.start_microphone() # runs microphone, saves recordings, returns file number
    