import numpy as np
from scipy import signal
import librosa
from typing import Optional, Tuple, Dict, List
import logging
from dataclasses import dataclass
import sounddevice as sd

@dataclass
class AudioMetrics:
    """Container for audio analysis metrics"""
    peak_amplitude: float
    rms_level: float
    crest_factor: float
    zero_crossings: int
    pitch: Optional[float]
    volume_db: float
    is_clipping: bool
    frequency_peaks: List[float]

class AudioProcessor:
    """Handles audio processing and analysis"""
    
    def __init__(self, sample_rate: int = 44100):
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        
    def analyze_audio(self, audio_data: np.ndarray) -> AudioMetrics:
        """Perform comprehensive audio analysis"""
        try:
            # Basic metrics
            peak = np.max(np.abs(audio_data))
            rms = np.sqrt(np.mean(np.square(audio_data)))
            crest_factor = peak / rms if rms > 0 else 0
            zero_crossings = np.sum(np.diff(np.signbit(audio_data)))
            
            # Volume in dB
            with np.errstate(divide='ignore', invalid='ignore'):
                volume_db = 20 * np.log10(rms)
                if not np.isfinite(volume_db):
                    volume_db = -np.inf
                    
            # Pitch estimation
            pitch = self._estimate_pitch(audio_data)
            
            # Frequency analysis
            freq_peaks = self._find_frequency_peaks(audio_data)
            
            # Clipping detection
            is_clipping = peak > 0.99
            
            return AudioMetrics(
                peak_amplitude=peak,
                rms_level=rms,
                crest_factor=crest_factor,
                zero_crossings=zero_crossings,
                pitch=pitch,
                volume_db=volume_db,
                is_clipping=is_clipping,
                frequency_peaks=freq_peaks
            )
            
        except Exception as e:
            self.logger.error(f"Audio analysis failed: {str(e)}")
            raise
            
    def normalize_audio(self, audio_data: np.ndarray,
                       target_level: float = -3.0) -> np.ndarray:
        """Normalize audio to target RMS level in dB"""
        try:
            rms = np.sqrt(np.mean(np.square(audio_data)))
            if rms > 0:
                current_db = 20 * np.log10(rms)
                gain = 10**((target_level - current_db) / 20)
                return np.clip(audio_data * gain, -1.0, 1.0)
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Normalization failed: {str(e)}")
            raise
            
    def remove_noise(self, audio_data: np.ndarray,
                    noise_threshold: float = -60.0) -> np.ndarray:
        """Remove noise using spectral gating"""
        try:
            # Convert threshold from dB to linear
            threshold = 10**(noise_threshold/20)
            
            # Compute spectrogram
            f, t, Sxx = signal.spectrogram(audio_data, fs=self.sample_rate)
            
            # Create noise mask
            mask = Sxx > threshold
            
            # Apply mask
            Sxx_clean = Sxx * mask
            
            # Reconstruct signal
            audio_clean = signal.istft(Sxx_clean, fs=self.sample_rate)[1]
            
            return audio_clean
            
        except Exception as e:
            self.logger.error(f"Noise removal failed: {str(e)}")
            raise
            
    def apply_effects(self, audio_data: np.ndarray,
                     effects: Dict[str, float]) -> np.ndarray:
        """Apply various audio effects"""
        try:
            processed_audio = audio_data.copy()
            
            # Apply gain
            if 'gain_db' in effects:
                processed_audio = self.apply_gain(processed_audio,
                                               effects['gain_db'])
            
            # Apply compression
            if 'compression' in effects:
                processed_audio = self.apply_compression(
                    processed_audio,
                    threshold=effects.get('comp_threshold', -20),
                    ratio=effects.get('comp_ratio', 4)
                )
            
            # Apply EQ
            if 'eq' in effects:
                processed_audio = self.apply_eq(
                    processed_audio,
                    effects['eq']
                )
                
            return processed_audio
            
        except Exception as e:
            self.logger.error(f"Effect processing failed: {str(e)}")
            raise
            
    def apply_gain(self, audio_data: np.ndarray, gain_db: float) -> np.ndarray:
        """Apply gain in decibels"""
        try:
            return np.clip(audio_data * (10 ** (gain_db / 20)), -1.0, 1.0)
        except Exception as e:
            self.logger.error(f"Gain application failed: {str(e)}")
            raise
            
    def apply_compression(self, audio_data: np.ndarray,
                         threshold: float = -20.0,
                         ratio: float = 4.0,
                         attack: float = 0.005,
                         release: float = 0.1) -> np.ndarray:
        """Apply dynamic range compression"""
        try:
            # Convert threshold from dB to linear
            threshold_lin = 10 ** (threshold / 20)
            
            # Calculate gain reduction
            gain_reduction = np.zeros_like(audio_data)
            envelope = np.zeros_like(audio_data)
            
            # Time constants
            attack_coeff = np.exp(-1 / (self.sample_rate * attack))
            release_coeff = np.exp(-1 / (self.sample_rate * release))
            
            # Process samples
            for i in range(len(audio_data)):
                # Envelope follower
                level = np.abs(audio_data[i])
                if level > envelope[i-1]:
                    envelope[i] = attack_coeff * (envelope[i-1] - level) + level
                else:
                    envelope[i] = release_coeff * (envelope[i-1] - level) + level
                
                # Gain computer
                if envelope[i] > threshold_lin:
                    gain_reduction[i] = (threshold_lin + 
                                      (envelope[i] - threshold_lin) / ratio) / envelope[i]
                else:
                    gain_reduction[i] = 1.0
                    
            return audio_data * gain_reduction
            
        except Exception as e:
            self.logger.error(f"Compression failed: {str(e)}")
            raise
            
    def apply_eq(self, audio_data: np.ndarray,
                bands: Dict[str, float]) -> np.ndarray:
        """Apply multi-band equalizer"""
        try:
            # Define frequency bands
            band_ranges = {
                'low': (20, 250),
                'mid': (250, 4000),
                'high': (4000, 20000)
            }
            
            processed_audio = np.zeros_like(audio_data)
            
            for band, gain_db in bands.items():
                if band in band_ranges:
                    low, high = band_ranges[band]
                    
                    # Design band-pass filter
                    b, a = signal.butter(4, 
                                       [low/(self.sample_rate/2),
                                        high/(self.sample_rate/2)],
                                       btype='band')
                    
                    # Filter and apply gain
                    filtered = signal.filtfilt(b, a, audio_data)
                    processed_audio += filtered * (10 ** (gain_db / 20))
                    
            return np.clip(processed_audio, -1.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"EQ processing failed: {str(e)}")
            raise
            
    def _estimate_pitch(self, audio_data: np.ndarray) -> Optional[float]:
        """Estimate fundamental frequency using zero-crossing rate"""
        try:
            # Use librosa's pitch detection
            pitches, magnitudes = librosa.piptrack(
                y=audio_data,
                sr=self.sample_rate,
                fmin=50,
                fmax=2000
            )
            
            # Get the highest magnitude pitch
            pit = pitches[magnitudes > 0.1]
            if len(pit) > 0:
                return float(np.median(pit))
            return None
            
        except Exception as e:
            self.logger.error(f"Pitch estimation failed: {str(e)}")
            return None
            
    def _find_frequency_peaks(self, audio_data: np.ndarray,
                            n_peaks: int = 5) -> List[float]:
        """Find dominant frequencies in the audio"""
        try:
            # Compute FFT
            fft = np.fft.rfft(audio_data)
            freqs = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
            magnitudes = np.abs(fft)
            
            # Find peaks
            peak_indices = signal.find_peaks(magnitudes)[0]
            sorted_peaks = sorted(zip(magnitudes[peak_indices],
                                    freqs[peak_indices]),
                                reverse=True)
            
            return [freq for mag, freq in sorted_peaks[:n_peaks]]
            
        except Exception as e:
            self.logger.error(f"Frequency peak detection failed: {str(e)}")
            return []
            
    def get_level_meter_data(self, audio_data: np.ndarray,
                            segment_size: int = 1024) -> np.ndarray:
        """Calculate level meter values for visualization"""
        try:
            segments = np.array_split(audio_data,
                                    max(1, len(audio_data) // segment_size))
            
            levels = np.array([np.sqrt(np.mean(np.square(s))) for s in segments])
            
            with np.errstate(divide='ignore', invalid='ignore'):
                db_levels = 20 * np.log10(levels)
                db_levels = np.clip(db_levels, -60, 0)
                
            return db_levels
            
        except Exception as e:
            self.logger.error(f"Level meter calculation failed: {str(e)}")
            raise
            
    def detect_silence(self, audio_data: np.ndarray,
                      threshold_db: float = -60.0,
                      min_duration: float = 0.1) -> List[Tuple[int, int]]:
        """Detect silent regions in audio"""
        try:
            # Calculate RMS energy
            frame_length = int(min_duration * self.sample_rate)
            hop_length = frame_length // 2
            
            rms = librosa.feature.rms(
                y=audio_data,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]
            
            # Convert to dB
            db = librosa.amplitude_to_db(rms)
            
            # Find silent regions
            silent = db < threshold_db
            
            # Convert frame indices to sample indices
            silent_regions = []
            current_start = None
            
            for i, is_silent in enumerate(silent):
                if is_silent and current_start is None:
                    current_start = i * hop_length
                elif not is_silent and current_start is not None:
                    silent_regions.append(
                        (current_start, i * hop_length)
                    )
                    current_start = None
                    
            return silent_regions
            
        except Exception as e:
            self.logger.error(f"Silence detection failed: {str(e)}")
            return []