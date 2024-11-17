import os

def save_transcription(filename, transcript):
    output_filename = os.path.splitext(filename)[0] + "_transcription.txt"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(transcript)
    return output_filename
