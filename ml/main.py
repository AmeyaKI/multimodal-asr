from a_detect.mic_vad import MicVAD
from silero_vad import load_silero_vad
from b_asr.asr_class import ASR
from c_llm.ollama_llm import query
from d_tts.tts import TTS

def main(): 
    # load Voice Activity Detection (VAD) model (Silero) and Mic (Sounddevice)
    vad = load_silero_vad() # 
    micvad_model = MicVAD(vad) 
    
    # load ASR (Default: Parakeet) 
    asr_model = ASR()
    
    # load TTS (Default: Bark)
    tts_model = TTS()
    
    user_audio_arr = micvad_model.start_microphone() # runs microphone, saves recordings, returns file number
    
    user_speech_to_text = asr_model.process_arr(user_audio_arr) # translates audio arr to text
    
    llm_response = query(user_speech_to_text) # queries LLM with user
    
    tts_model.generate_speech(llm_response)()


if __name__ == '__main__':
    main()