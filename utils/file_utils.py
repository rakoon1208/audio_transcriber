import os
from pathlib import Path
from datetime import datetime
import json
import shutil
import logging
from typing import Optional, Dict, List, Union
import wave
import numpy as np
import soundfile as sf

class FileManager:
    """Manages file operations for recordings and transcriptions"""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize FileManager with a base directory.
        If no directory is specified, creates one in user's home directory.
        """
        self.logger = logging.getLogger(__name__)
        
        # Set up base directory structure
        self.base_dir = Path(base_dir) if base_dir else Path.home() / "audio_recordings"
        self.recordings_dir = self.base_dir / "recordings"
        self.transcripts_dir = self.base_dir / "transcripts"
        self.temp_dir = self.base_dir / "temp"
        self.backup_dir = self.base_dir / "backups"
        
        # Create directory structure
        self._setup_directories()
        
    def _setup_directories(self) -> None:
        """Create all necessary directories if they don't exist"""
        try:
            for directory in [self.recordings_dir, self.transcripts_dir, 
                            self.temp_dir, self.backup_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Directory structure created at {self.base_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create directories: {str(e)}")
            raise

    def get_save_path(self, filename: str, directory: str = "recordings") -> Path:
        """Get the full path for saving a file"""
        if not filename.endswith(('.wav', '.txt', '.json')):
            filename += '.wav'  # Default to WAV for audio files
            
        dir_map = {
            "recordings": self.recordings_dir,
            "transcripts": self.transcripts_dir,
            "temp": self.temp_dir,
            "backups": self.backup_dir
        }
        
        return dir_map.get(directory, self.recordings_dir) / filename

    def save_audio(self, audio_data: Union[bytes, np.ndarray], 
                  filename: str, 
                  sample_rate: int = 44100,
                  metadata: Optional[Dict] = None) -> Path:
        """
        Save audio data to file with optional metadata
        """
        try:
            filepath = self.get_save_path(filename)
            
            # Save audio file
            if isinstance(audio_data, bytes):
                with open(filepath, 'wb') as f:
                    f.write(audio_data)
            else:
                sf.write(str(filepath), audio_data, sample_rate)
            
            # Save metadata if provided
            if metadata:
                self.save_metadata(filepath, metadata)
                
            self.logger.info(f"Saved audio to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save audio: {str(e)}")
            raise

    def save_metadata(self, audio_path: Path, metadata: Dict) -> Path:
        """Save metadata for an audio file"""
        try:
            meta_path = audio_path.with_suffix('.json')
            metadata_with_timestamp = {
                'timestamp': datetime.now().isoformat(),
                'audio_file': audio_path.name,
                **metadata
            }
            
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_with_timestamp, f, indent=2)
                
            return meta_path
            
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {str(e)}")
            raise

    def save_transcript(self, text: str, audio_filename: str, 
                       format: str = 'txt') -> Path:
        """
        Save transcription in specified format
        Supported formats: txt, json, srt
        """
        try:
            base_name = Path(audio_filename).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format == 'txt':
                filepath = self.transcripts_dir / f"{base_name}_transcript_{timestamp}.txt"
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(text)
                    
            elif format == 'json':
                filepath = self.transcripts_dir / f"{base_name}_transcript_{timestamp}.json"
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump({'transcript': text, 'timestamp': timestamp}, f, indent=2)
                    
            elif format == 'srt':
                filepath = self.transcripts_dir / f"{base_name}_transcript_{timestamp}.srt"
                self._save_as_srt(text, filepath)
                
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            self.logger.info(f"Saved transcript to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save transcript: {str(e)}")
            raise

    def _save_as_srt(self, text: str, filepath: Path) -> None:
        """Save transcript in SRT subtitle format"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                segments = text.split('\n\n')
                for i, segment in enumerate(segments, 1):
                    if segment.strip():
                        f.write(f"{i}\n")
                        f.write("00:00:00,000 --> 00:00:00,000\n")  # Placeholder timings
                        f.write(f"{segment.strip()}\n\n")
        except Exception as e:
            self.logger.error(f"Failed to save SRT: {str(e)}")
            raise

    def list_recordings(self, sort_by: str = 'date') -> List[Dict]:
        """
        List all recordings with their metadata
        sort_by options: 'date', 'name', 'duration'
        """
        recordings = []
        try:
            for filepath in self.recordings_dir.glob('*.wav'):
                info = self.get_audio_info(filepath)
                metadata = self.get_metadata(filepath)
                recordings.append({
                    'path': filepath,
                    'filename': filepath.name,
                    'info': info,
                    'metadata': metadata
                })
            
            if sort_by == 'date':
                recordings.sort(key=lambda x: x['path'].stat().st_mtime, reverse=True)
            elif sort_by == 'name':
                recordings.sort(key=lambda x: x['filename'])
            elif sort_by == 'duration':
                recordings.sort(key=lambda x: x['info'].get('duration', 0), reverse=True)
                
            return recordings
            
        except Exception as e:
            self.logger.error(f"Failed to list recordings: {str(e)}")
            return []

    def get_audio_info(self, filepath: Path) -> Dict:
        """Get technical information about an audio file"""
        try:
            with wave.open(str(filepath), 'rb') as wav:
                return {
                    'channels': wav.getnchannels(),
                    'sample_width': wav.getsampwidth(),
                    'frame_rate': wav.getframerate(),
                    'n_frames': wav.getnframes(),
                    'duration': wav.getnframes() / wav.getframerate()
                }
        except Exception as e:
            self.logger.error(f"Failed to get audio info: {str(e)}")
            return {}

    def get_metadata(self, filepath: Path) -> Optional[Dict]:
        """Get metadata for an audio file"""
        try:
            meta_path = filepath.with_suffix('.json')
            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            self.logger.error(f"Failed to read metadata: {str(e)}")
            return None

    def create_backup(self, filepath: Path) -> Path:
        """Create a backup copy of a file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"{filepath.stem}_backup_{timestamp}{filepath.suffix}"
            shutil.copy2(filepath, backup_path)
            self.logger.info(f"Created backup at {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"Failed to create backup: {str(e)}")
            raise

    def cleanup_temp_files(self, max_age_days: int = 7) -> None:
        """Remove temporary files older than specified days"""
        try:
            cutoff = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
            for item in self.temp_dir.iterdir():
                if item.stat().st_mtime < cutoff:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            self.logger.info("Cleaned up temporary files")
        except Exception as e:
            self.logger.error(f"Failed to cleanup temp files: {str(e)}")
            raise

    def delete_recording(self, filepath: Path, create_backup: bool = True) -> None:
        """Delete a recording and its associated files"""
        try:
            if create_backup:
                self.create_backup(filepath)
                
            # Delete main audio file
            filepath.unlink()
            
            # Delete associated files
            meta_path = filepath.with_suffix('.json')
            if meta_path.exists():
                meta_path.unlink()
                
            # Delete associated transcripts
            base_name = filepath.stem
            for transcript in self.transcripts_dir.glob(f"{base_name}_transcript_*"):
                transcript.unlink()
                
            self.logger.info(f"Deleted recording: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to delete recording: {str(e)}")
            raise