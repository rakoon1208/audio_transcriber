import whisper
import torch
import numpy as np
import threading
import queue
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
import logging
from pathlib import Path

@dataclass
class TranscriptionConfig:
    """Configuration for transcription"""
    model_size: str = "medium"
    language: Optional[str] = None
    task: str = "transcribe"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type: str = "float16" if torch.cuda.is_available() else "float32"
    batch_size: int = 8

class TranscriptionManager:
    """Manages audio transcription using Whisper"""
    
    def __init__(self, config: Optional[TranscriptionConfig] = None):
        self.config = config or TranscriptionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize state
        self.model = None
        self._transcription_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._is_processing = False
        
        # Threading
        self._process_thread = None
        self._lock = threading.Lock()
    
    def initialize_model(self) -> bool:
        """Initialize the Whisper model"""
        try:
            if self.model:
                del self.model
                torch.cuda.empty_cache()
            
            self.model = whisper.load_model(
                self.config.model_size,
                device=self.config.device
            )
            
            if self.config.device == "cuda":
                self.model = self.model.to(torch.float16)
            
            self.logger.info(f"Initialized Whisper model: {self.config.model_size}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            return False
    
    def start_processing(self) -> None:
        """Start the transcription processing thread"""
        if self._is_processing:
            return
            
        if not self.model:
            self.initialize_model()
        
        self._is_processing = True
        self._process_thread = threading.Thread(target=self._process_queue)
        self._process_thread.daemon = True
        self._process_thread.start()
    
    def stop_processing(self) -> None:
        """Stop the transcription processing"""
        self._is_processing = False
        if self._process_thread:
            self._process_thread.join()
    
    def transcribe_audio(self, audio_data: np.ndarray,
                        callback: Optional[Callable[[Dict], None]] = None) -> None:
        """Queue audio data for transcription"""
        self._transcription_queue.put((audio_data, callback))
        if not self._is_processing:
            self.start_processing()
    
    def _process_queue(self) -> None:
        """Process queued transcription requests"""
        while self._is_processing:
            try:
                audio_data, callback = self._transcription_queue.get(timeout=1.0)
                result = self._transcribe(audio_data)
                if callback:
                    callback(result)
                self._result_queue.put(result)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Transcription error: {str(e)}")
    
    def _transcribe(self, audio_data: np.ndarray) -> Dict:
        """Perform transcription on audio data"""
        try:
            with self._lock:
                result = self.model.transcribe(
                    audio_data,
                    language=self.config.language,
                    task=self.config.task,
                    fp16=(self.config.compute_type == "float16")
                )
                
                # Process and format result
                processed_result = {
                    'text': result['text'],
                    'segments': [
                        {
                            'text': seg['text'],
                            'start': seg['start'],
                            'end': seg['end'],
                            'confidence': seg['confidence']
                        }
                        for seg in result['segments']
                    ],
                    'language': result['language']
                }
                
                return processed_result
                
        except Exception as e:
            self.logger.error(f"Transcription processing error: {str(e)}")
            raise
    
    def get_results(self) -> List[Dict]:
        """Get all available transcription results"""
        results = []
        while not self._result_queue.empty():
            results.append(self._result_queue.get())
        return results
    
    def clear_queues(self) -> None:
        """Clear all transcription queues"""
        while not self._transcription_queue.empty():
            self._transcription_queue.get()
        while not self._result_queue.empty():
            self._result_queue.get()