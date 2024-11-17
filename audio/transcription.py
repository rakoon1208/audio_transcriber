import torch
import whisper
import os

def transcribe_audio_file(filename):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("medium").to(device)
    result = model.transcribe(filename, fp16=True)
    transcript = result.get("text", "").strip()
    output_filename = os.path.splitext(filename)[0] + "_transcription.txt"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(transcript)
    
    return output_filename
