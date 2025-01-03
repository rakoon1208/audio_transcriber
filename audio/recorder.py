import sounddevice as sd
import numpy as np
import wave
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Optional, Callable

@dataclass
class AudioConfig:
    """Configuration settings for audio recording"""
    sample_rate: int = 44100
    channels: int = 1
    dtype: str = 'float32'
    blocksize: int = 1024
    device: Optional[int] = None

class AudioRecorder:
    """Handles audio recording functionality"""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.logger = logging.getLogger(__name__)
        self.recording = False
        self.paused = False
        self._audio_buffer = []
        self.stream = None
        self.current_device = None
        
    def set_device(self, device_id: int) -> None:
        """Set the recording device"""
        self.current_device = device_id
        self.logger.info(f"Set recording device to: {device_id}")
        
    def start_recording(self, callback: Optional[Callable] = None) -> None:
        """Start the recording process"""
        if self.recording:
            self.logger.warning("Recording already in progress")
            return
            
        def internal_callback(indata, frames, time_info, status):
            """Internal callback to handle incoming audio data"""
            if status:
                self.logger.warning(f"Recording callback status: {status}")
            if not self.paused:
                self._audio_buffer.append(indata.copy())
            if callback:
                callback(indata, frames, time_info, status)
                
        try:
            self.recording = True
            self._audio_buffer = []
            self.stream = sd.InputStream(
                device=self.current_device,
                channels=self.config.channels,
                samplerate=self.config.sample_rate,
                callback=internal_callback,
                dtype=self.config.dtype,
                blocksize=self.config.blocksize
            )
            self.stream.start()
            self.logger.info("Recording started")
        except Exception as e:
            self.recording = False
            self.logger.error(f"Failed to start recording: {str(e)}")
            raise
            
    def stop_recording(self) -> None:
        """Stop the recording process"""
        if not self.recording:
            return
            
        try:
            self.recording = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            self.logger.info("Recording stopped")
        except Exception as e:
            self.logger.error(f"Error stopping recording: {str(e)}")
            raise
            
    def pause_recording(self) -> None:
        """Pause the recording"""
        self.paused = True
        self.logger.info("Recording paused")
        
    def resume_recording(self) -> None:
        """Resume the recording"""
        self.paused = False
        self.logger.info("Recording resumed")
        
    def save_recording(self, filename: str) -> Path:
        """Save the recorded audio to a WAV file"""
        if not self._audio_buffer:
            raise ValueError("No audio data available to save")
            
        try:
            # Ensure filename has .wav extension
            if not filename.endswith('.wav'):
                filename += '.wav'
                
            filepath = Path(filename)
            
            # Combine all audio chunks
            audio_data = np.concatenate(self._audio_buffer)
            
            # Convert to int16 format
            audio_int16 = np.int16(audio_data * 32767)
            
            # Save as WAV file
            with wave.open(str(filepath), 'wb') as wf:
                wf.setnchannels(self.config.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.config.sample_rate)
                wf.writeframes(audio_int16.tobytes())
                
            self.logger.info(f"Recording saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save recording: {str(e)}")
            raise
            
    @property
    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self.recording
        
    @property
    def is_paused(self) -> bool:
        """Check if recording is paused"""
        return self.paused