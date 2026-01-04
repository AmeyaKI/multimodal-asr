from a_detect.mic_vad import MicVAD
from silero_vad import load_silero_vad
from b_asr.asr_class import ASR
from c_llm.ollama_llm import query

def main(): 
    # load Voice Activity Detection (VAD) model (Silero) and Mic (Sounddevice)
    vad_model = load_silero_vad() # 
    micvad_model = MicVAD(vad_model) 
    
    # load ASR (Default: Parakeet) 
    asr_model = ASR()
    
    user_audio_arr = micvad_model.start_microphone() # runs microphone, saves recordings, returns file number
    
    user_speech_to_text = asr_model.process_arr(user_audio_arr) # translates audio arr to text
    
    llm_response = query(user_speech_to_text) # queries LLM with user


if __name__ == '__main__':
    main()