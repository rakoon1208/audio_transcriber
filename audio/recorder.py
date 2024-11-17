import sounddevice as sd
import numpy as np
import wave
from datetime import datetime

class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.audio_data = []
        self.sample_rate = 16000
        self.channels = 1
        self.stream = None
        self.current_device = None

    def set_device(self, device_index):
        self.current_device = device_index

    def start_recording(self, callback):
        # Check if the callback is provided; otherwise, use the default behavior
        def internal_callback(indata, frames, time_info, status):
            self.audio_data.append(indata.copy())
            if callback:
                callback(indata, frames, time_info, status)  

        self.recording = True
        self.audio_data = []

        self.stream = sd.InputStream(
            device=self.current_device,
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=internal_callback,
            dtype='float32'
        )
        self.stream.start()

    def stop_recording(self):
        self.recording = False
        self.stream.stop()
        self.stream.close()

    def is_recording(self):
        return self.recording

    def get_audio_data(self):
        if self.audio_data:
            return np.concatenate(self.audio_data).flatten()
        return None

    def save_recording(self, filename=None):
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}"
        filename = f"{filename}.wav"

        audio_array = np.concatenate([chunk.flatten() for chunk in self.audio_data])
        audio_int16 = np.int16(audio_array * 32767)

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
