import sounddevice as sd

def get_available_devices():
    return [device for device in sd.query_devices() if device['max_input_channels'] > 0]
