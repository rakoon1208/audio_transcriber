import sounddevice as sd
import numpy as np
import wave
import threading
import queue
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Optional, Callable, Any

@dataclass
class AudioConfig:
    """Configuration for audio recording"""
    sample_rate: int = 44100
    channels: int = 1
    dtype: str = 'float32'
    blocksize: int = 1024
    device: Optional[int] = None
    latency: str = 'low'

class AudioRecorder:
    """Handles audio recording with real-time processing capabilities"""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.recording = False
        self.paused = False
        self._audio_queue = queue.Queue()
        self._audio_buffer = []
        
        # Stream management
        self.stream = None
        self._lock = threading.Lock()
        
    def start_recording(self, callback: Optional[Callable[[np.ndarray, Any], None]] = None):
        """Start audio recording with optional real-time callback"""
        if self.recording:
            self.logger.warning("Recording already in progress")
            return
        
        try:
            def audio_callback(indata, frames, time, status):
                if status:
                    self.logger.warning(f"Audio callback status: {status}")
                if not self.paused:
                    self._audio_buffer.append(indata.copy())
                    if callback:
                        callback(indata, time)
            
            self.stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                blocksize=self.config.blocksize,
                device=self.config.device,
                latency=self.config.latency,
                callback=audio_callback
            )
            
            self.recording = True
            self.stream.start()
            self.logger.info("Recording started")
            
        except Exception as e:
            self.logger.error(f"Failed to start recording: {str(e)}")
            raise
    
    def stop_recording(self) -> None:
        """Stop the recording"""
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
        with self._lock:
            self.paused = True
            self.logger.info("Recording paused")
    
    def resume_recording(self) -> None:
        """Resume the recording"""
        with self._lock:
            self.paused = False
            self.logger.info("Recording resumed")
    
    def get_audio_data(self) -> Optional[np.ndarray]:
        """Get the recorded audio data"""
        if not self._audio_buffer:
            return None
        return np.concatenate(self._audio_buffer)
    
    def save_recording(self, filename: Optional[str] = None) -> Path:
        """Save the recording to a WAV file"""
        if not self._audio_buffer:
            raise ValueError("No audio data available to save")
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
            
        filepath = Path(filename)
        if not filepath.suffix:
            filepath = filepath.with_suffix('.wav')
            
        try:
            audio_data = np.concatenate(self._audio_buffer)
            
            # Convert to int16 for WAV file
            audio_int16 = np.int16(audio_data * 32767)
            
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
    
    def clear_buffer(self) -> None:
        """Clear the audio buffer"""
        with self._lock:
            self._audio_buffer.clear()
            self.logger.info("Audio buffer cleared")
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self.recording
    
    @property
    def is_paused(self) -> bool:
        """Check if recording is paused"""
        return self.paused