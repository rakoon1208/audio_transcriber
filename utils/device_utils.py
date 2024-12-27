import sounddevice as sd
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass

@dataclass
class AudioDevice:
    """Represents an audio input device"""
    id: int
    name: str
    channels: int
    sample_rates: List[int]
    default_sample_rate: int
    is_default: bool

class DeviceManager:
    """Manages audio devices and their configurations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._update_devices()
        
    def _update_devices(self):
        """Update the list of available devices"""
        try:
            self.devices = []
            default_device = sd.default.device[0]  # Get default input device
            
            for idx, device in enumerate(sd.query_devices()):
                if device['max_input_channels'] > 0:  # Only input devices
                    self.devices.append(AudioDevice(
                        id=idx,
                        name=device['name'],
                        channels=device['max_input_channels'],
                        sample_rates=self._get_supported_rates(device),
                        default_sample_rate=int(device['default_samplerate']),
                        is_default=(idx == default_device)
                    ))
                    
        except Exception as e:
            self.logger.error(f"Error updating devices: {str(e)}")
            raise
            
    def _get_supported_rates(self, device: Dict) -> List[int]:
        """Get supported sample rates for a device"""
        standard_rates = [44100, 48000, 96000]
        supported_rates = []
        
        for rate in standard_rates:
            try:
                sd.check_input_settings(
                    device=device['name'],
                    samplerate=rate
                )
                supported_rates.append(rate)
            except:
                continue
                
        return supported_rates
    
    def get_devices(self) -> List[AudioDevice]:
        """Get list of available input devices"""
        self._update_devices()
        return self.devices
    
    def get_default_device(self) -> Optional[AudioDevice]:
        """Get the default input device"""
        for device in self.devices:
            if device.is_default:
                return device
        return self.devices[0] if self.devices else None
    
    def test_device(self, device_id: int) -> bool:
        """Test if a device is available and working"""
        try:
            sd.check_input_settings(
                device=device_id,
                samplerate=44100,
                channels=1
            )
            return True
        except Exception as e:
            self.logger.warning(f"Device test failed: {str(e)}")
            return False
    
    def get_device_by_id(self, device_id: int) -> Optional[AudioDevice]:
        """Get device by its ID"""
        for device in self.devices:
            if device.id == device_id:
                return device
        return None
    
    def get_optimal_settings(self, device_id: int) -> Dict:
        """Get optimal audio settings for a device"""
        device = self.get_device_by_id(device_id)
        if not device:
            raise ValueError(f"No device found with ID {device_id}")
            
        return {
            'sample_rate': device.default_sample_rate,
            'channels': min(2, device.channels),  # Prefer stereo or mono
            'dtype': 'float32',
            'latency': 'low'
        }