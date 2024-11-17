import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from utils.device_utils import get_available_devices
from audio.recorder import AudioRecorder
from audio.transcription import transcribe_audio_file
from gui.plot import AudioPlot
import queue
import numpy as np

class AudioRecorderGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Audio Recorder & File Transcriber")
        self.root.geometry("900x700")
        self.root.resizable(False, False)

        self.audio_recorder = AudioRecorder()
        self.plot = AudioPlot(self.root)
        self.audio_queue = queue.Queue() 
        self._init_gui()
    
    def _init_gui(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10, padx=10, fill=tk.X)
        
        device_frame = ttk.LabelFrame(control_frame, text="Input Device")
        device_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.device_var = tk.StringVar()
        device_menu = ttk.Combobox(device_frame, textvariable=self.device_var, state="readonly", width=50)
        self.devices = get_available_devices()
        device_menu['values'] = [f"{i}: {d['name']}" for i, d in enumerate(self.devices)]
        device_menu.set(f"{self.devices[0]['name']}")
        device_menu.bind('<<ComboboxSelected>>', self.change_device)
        device_menu.pack(padx=5, pady=5)
        
        self.record_button = ttk.Button(control_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=20)

        trans_frame = ttk.LabelFrame(self.root, text="File Transcription")
        trans_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        upload_button = ttk.Button(trans_frame, text="Upload WAV File", command=self.upload_wav)
        upload_button.pack(pady=10)
        
        self.status_label = ttk.Label(trans_frame, text="")
        self.status_label.pack(pady=5)

    def change_device(self, event):
        selected = self.device_var.get()
        self.audio_recorder.set_device(int(selected.split(':')[0]))
        
    def toggle_recording(self):
        if not self.audio_recorder.is_recording():
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        # Start recording and initiate audio plot 
        self.audio_recorder.start_recording(callback=self.audio_callback)
        self.record_button.config(text="Stop Recording")
        self.update_plot()  # Start the update loop for the plot

    def stop_recording(self):
        self.audio_recorder.stop_recording()
        self.record_button.config(text="Start Recording")
        self.save_recording_as()  # Prompt to save the file

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f'Warning: {status}')
        self.audio_queue.put(indata.flatten())  # Flatten and add to queue for plotting

    def update_plot(self):
        try:
            while not self.audio_queue.empty():
                data = self.audio_queue.get()
                normalized_data = data / np.max(np.abs(data)) if np.max(np.abs(data)) != 0 else data
                update_size = len(normalized_data)
                self.plot.plot_data = np.roll(self.plot.plot_data, -update_size)
                self.plot.plot_data[-update_size:] = normalized_data
                self.plot.line.set_ydata(self.plot.plot_data)

            self.plot.canvas.draw()

            if self.audio_recorder.is_recording():
                self.root.after(50, self.update_plot)  
        except Exception as e:
            messagebox.showerror("Error", f"Plot update failed: {str(e)}")

    def save_recording_as(self):
        filename = simpledialog.askstring("Save Recording", "Enter filename for the recording:")
        if filename:
            self.audio_recorder.save_recording(filename)
            messagebox.showinfo("Recording Saved", f"Recording saved as {filename}.wav")
    
    def upload_wav(self):
        filename = filedialog.askopenfilename(title="Select WAV File", filetypes=[("WAV files", "*.wav")])
        if filename:
            self.status_label.config(text="Transcribing... Please wait.")
            self.root.update()
            transcript = transcribe_audio_file(filename)
            self.status_label.config(text=f"Transcription saved to: {transcript}")

    def run(self):
        self.root.mainloop()
