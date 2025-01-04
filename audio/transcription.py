import whisper
import torch
import numpy as np
import threading
import queue
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import soundfile as sf

@dataclass
class TranscriptionConfig:
    """Configuration for transcription"""
    model_size: str = "tiny"  # Options: tiny, base, small, medium
    language: Optional[str] = None
    chunk_duration: int = 30  # Process 30 seconds at a time
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type: str = "float16" if torch.cuda.is_available() else "float32"

class TranscriptionManager:
    """Manages audio transcription using Whisper"""
    
    def __init__(self, config: Optional[TranscriptionConfig] = None):
        self.config = config or TranscriptionConfig()
        self.logger = logging.getLogger(__name__)
        self.model = None
        
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
            
            self.logger.info(f"Initialized Whisper model: {self.config.model_size}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            return False

    def transcribe_audio(self, audio_path: str, callback: Optional[Callable] = None) -> Dict:
        """
        Transcribe audio file with progress updates
        
        Args:
            audio_path: Path to audio file
            callback: Optional callback function for progress updates
            
        Returns:
            Dictionary containing transcription results
        """
        try:
            # Initialize model if needed
            if not self.model:
                if not self.initialize_model():
                    raise RuntimeError("Failed to initialize transcription model")

            # Load audio file
            self.logger.info(f"Loading audio file: {audio_path}")
            audio, sr = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Calculate chunk size in samples
            chunk_samples = sr * self.config.chunk_duration
            
            # Split audio into chunks
            chunks = [audio[i:i + chunk_samples] 
                     for i in range(0, len(audio), chunk_samples)]
            
            full_transcription = []
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                try:
                    # Clear GPU memory before processing each chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Transcribe chunk
                    result = self.model.transcribe(
                        chunk,
                        language=self.config.language,
                        fp16=(self.config.compute_type == "float16")
                    )
                    
                    # Add to full transcription
                    full_transcription.append(result['text'])
                    
                    # Update progress
                    if callback:
                        progress = {
                            'progress': (i + 1) / len(chunks),
                            'text': result['text']
                        }
                        callback(progress)
                        
                except Exception as e:
                    self.logger.error(f"Chunk transcription failed: {str(e)}")
                    continue

            # Combine results
            final_result = {
                'text': ' '.join(full_transcription),
                'language': self.config.language or 'auto',
                'chunks_processed': len(chunks),
                'model_size': self.config.model_size
            }

            self.logger.info("Transcription completed successfully")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            raise

    def cleanup(self):
        """Clean up resources and free memory"""
        try:
            if hasattr(self, 'model') and self.model:
                del self.model
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.logger.info("Transcription manager cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")

    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        return {
            'model_size': self.config.model_size,
            'device': self.config.device,
            'compute_type': self.config.compute_type,
            'is_gpu_available': torch.cuda.is_available(),
            'model_loaded': self.model is not None
        }

    def set_model_size(self, size: str) -> bool:
        """
        Change the model size and reinitialize
        Args:
            size: One of 'tiny', 'base', 'small', 'medium'
        """
        if size not in ['tiny', 'base', 'small', 'medium']:
            self.logger.error(f"Invalid model size: {size}")
            return False
            
        try:
            self.cleanup()  # Clean up existing model
            self.config.model_size = size
            return self.initialize_model()
        except Exception as e:
            self.logger.error(f"Failed to set model size: {str(e)}")
            return False

    def is_ready(self) -> bool:
        """Check if the transcription manager is ready to process audio"""
        return self.model is not None or self.initialize_model()