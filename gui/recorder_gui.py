import tkinter as tk
from tkinter import ttk, messagebox
import logging
from datetime import datetime
from pathlib import Path
from . import visualization
from audio.recorder import AudioRecorder, AudioConfig
from audio.transcription import TranscriptionManager, TranscriptionConfig
from utils.device_utils import DeviceManager
from utils.file_utils import FileManager

class AudioRecorderGUI:
    def __init__(self):
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("Audio Recorder & Transcriber")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Initialize core components
        self.logger = logging.getLogger(__name__)
        self.audio_recorder = AudioRecorder()
        self.transcription_manager = TranscriptionManager()
        self.device_manager = DeviceManager()
        self.file_manager = FileManager()

        # Setup GUI components
        self._init_variables()
        self._init_gui()
        self._init_bindings()

    def _init_variables(self):
        """Initialize state variables"""
        self.recording_time = 0
        self.timer_id = None
        self.is_paused = False
        self.device_var = tk.StringVar()
        self.timer_var = tk.StringVar(value="00:00:00")
        self.status_var = tk.StringVar(value="Ready")
        self.transcription_progress = tk.StringVar(value="")
        self.last_recording_path = None

    def _init_gui(self):
        """Initialize the GUI layout"""
        # Create main container
        self.main_container = ttk.Frame(self.root, padding="10")
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Create GUI sections
        self._create_device_section()
        self._create_controls_section()
        self._create_visualization_section()
        self._create_transcription_section()
        self._create_status_bar()

    def _create_device_section(self):
        """Create device selection section"""
        device_frame = ttk.LabelFrame(self.main_container, text="Audio Device", padding="5")
        device_frame.pack(fill=tk.X, pady=(0, 10))

        # Device dropdown
        devices = self.device_manager.get_devices()
        self.device_combobox = ttk.Combobox(
            device_frame,
            textvariable=self.device_var,
            values=[f"{dev.id}: {dev.name}" for dev in devices],
            state="readonly",
            width=50
        )
        self.device_combobox.pack(side=tk.LEFT, padx=5)
        self.device_combobox.bind('<<ComboboxSelected>>', self._on_device_change)

        # Set default device if available
        if devices:
            default_device = next((dev for dev in devices if dev.is_default), devices[0])
            self.device_combobox.set(f"{default_device.id}: {default_device.name}")
            self.audio_recorder.set_device(default_device.id)

        # Refresh button
        ttk.Button(
            device_frame,
            text="↻",
            width=3,
            command=self._refresh_devices
        ).pack(side=tk.LEFT, padx=5)

    def _create_controls_section(self):
        """Create recording controls section"""
        controls_frame = ttk.LabelFrame(self.main_container, text="Controls", padding="5")
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        # Recording controls
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(side=tk.LEFT, padx=5)

        self.record_button = ttk.Button(
            button_frame,
            text="Start Recording",
            command=self._toggle_recording
        )
        self.record_button.pack(side=tk.LEFT, padx=2)

        self.pause_button = ttk.Button(
            button_frame,
            text="Pause",
            command=self._toggle_pause,
            state="disabled"
        )
        self.pause_button.pack(side=tk.LEFT, padx=2)

        # Timer display
        ttk.Label(
            controls_frame,
            textvariable=self.timer_var,
            font=("Arial", 14)
        ).pack(side=tk.LEFT, padx=20)

    def _create_visualization_section(self):
        """Create audio visualization section"""
        self.visualizer = visualization.AudioVisualizer(self.main_container)
        self.visualizer.frame.pack(fill=tk.X, pady=(0, 10))

    def _create_transcription_section(self):
        """Create transcription section"""
        trans_frame = ttk.LabelFrame(self.main_container, text="Transcription", padding="5")
        trans_frame.pack(fill=tk.BOTH, expand=True)

        # Transcription controls
        control_frame = ttk.Frame(trans_frame)
        control_frame.pack(fill=tk.X, pady=(0, 5))

        self.transcribe_button = ttk.Button(
            control_frame,
            text="Transcribe Recording",
            command=self._transcribe_last_recording,
            state="disabled"
        )
        self.transcribe_button.pack(side=tk.LEFT, padx=5)

        ttk.Label(
            control_frame,
            textvariable=self.transcription_progress
        ).pack(side=tk.LEFT, padx=5)

        # Transcription display
        text_frame = ttk.Frame(trans_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.transcript_text = tk.Text(
            text_frame,
            wrap=tk.WORD,
            height=10,
            font=("Arial", 10)
        )
        
        scrollbar = ttk.Scrollbar(text_frame, command=self.transcript_text.yview)
        self.transcript_text.configure(yscrollcommand=scrollbar.set)

        self.transcript_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_status_bar(self):
        """Create status bar"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            padding=(5, 2)
        ).pack(fill=tk.X)

    def _init_bindings(self):
        """Initialize keyboard shortcuts"""
        self.root.bind('<Control-r>', lambda e: self._toggle_recording())
        self.root.bind('<Control-p>', lambda e: self._toggle_pause())
        self.root.bind('<Control-s>', lambda e: self._save_recording())
        self.root.bind('<Control-t>', lambda e: self._transcribe_last_recording())
        self.root.bind('<Escape>', lambda e: self._stop_recording())

    def _toggle_recording(self):
        """Toggle recording state"""
        if not self.audio_recorder.is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        """Start recording"""
        try:
            self.audio_recorder.start_recording(callback=self._audio_callback)
            self.record_button.configure(text="Stop Recording")
            self.pause_button.configure(state="normal")
            self.transcribe_button.configure(state="disabled")
            self._start_timer()
            self._update_status("Recording...")
        except Exception as e:
            self._handle_error("Failed to start recording", e)

    def _stop_recording(self):
        """Stop recording"""
        try:
            self.audio_recorder.stop_recording()
            self.record_button.configure(text="Start Recording")
            self.pause_button.configure(state="disabled")
            self._stop_timer()
            self._save_recording()
            self.transcribe_button.configure(state="normal")
            self._update_status("Recording stopped")
        except Exception as e:
            self._handle_error("Failed to stop recording", e)

    def _toggle_pause(self):
        """Toggle pause state"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.audio_recorder.pause_recording()
            self.pause_button.configure(text="Resume")
            self._update_status("Recording paused")
        else:
            self.audio_recorder.resume_recording()
            self.pause_button.configure(text="Pause")
            self._update_status("Recording resumed")

    def _audio_callback(self, indata, frames, time_info, status):
        """Handle incoming audio data"""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        if indata is not None:
            self.visualizer.update(indata)

    def _save_recording(self):
        """Save the current recording"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.file_manager.get_save_path(f"recording_{timestamp}")
            path = self.audio_recorder.save_recording(str(filename))
            self.last_recording_path = path
            self._update_status(f"Recording saved: {path}")
        except Exception as e:
            self._handle_error("Failed to save recording", e)

    def _transcribe_last_recording(self):
        """Transcribe the last recording"""
        if not self.last_recording_path:
            messagebox.showwarning("Warning", "No recording available to transcribe")
            return

        self.transcribe_button.configure(state="disabled")
        self.transcript_text.delete(1.0, tk.END)
        self._transcribe_recording(self.last_recording_path)

    def _transcribe_recording(self, filepath):
        """Handle transcription of recording"""
        try:
            def progress_callback(info):
                progress = info.get('progress', 0)
                text = info.get('text', '')
                self.transcription_progress.set(f"Progress: {progress*100:.0f}%")
                self.transcript_text.insert(tk.END, text + " ")
                self.transcript_text.see(tk.END)
                self.root.update()

            self._update_status("Starting transcription...")
            result = self.transcription_manager.transcribe_audio(
                str(filepath),
                callback=progress_callback
            )
            
            self.transcription_progress.set("")
            self._update_status("Transcription complete")
            
        except Exception as e:
            self._handle_error("Transcription failed", e)
        finally:
            self.transcribe_button.configure(state="normal")

    def _start_timer(self):
        """Start the recording timer"""
        self.recording_time = 0
        self._update_timer()

    def _stop_timer(self):
        """Stop the recording timer"""
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
            self.timer_id = None

    def _update_timer(self):
        """Update the timer display"""
        if not self.is_paused:
            self.recording_time += 1
            hours = self.recording_time // 3600
            minutes = (self.recording_time % 3600) // 60
            seconds = self.recording_time % 60
            self.timer_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
        self.timer_id = self.root.after(1000, self._update_timer)

    def _update_status(self, message):
        """Update status bar message"""
        self.status_var.set(message)
        self.logger.info(message)

    def _handle_error(self, message, error):
        """Handle and display errors"""
        error_msg = f"{message}: {str(error)}"
        self.logger.error(error_msg)
        messagebox.showerror("Error", error_msg)
        self._update_status("Error occurred")

    def _refresh_devices(self):
        """Refresh the device list"""
        devices = self.device_manager.get_devices()
        self.device_var.set('')
        self.device_combobox['values'] = [f"{dev.id}: {dev.name}" for dev in devices]

    def _on_device_change(self, event):
        """Handle device selection change"""
        selected = self.device_var.get()
        device_id = int(selected.split(':')[0])
        self.audio_recorder.set_device(device_id)

    def _on_closing(self):
        """Handle application closing"""
        if self.audio_recorder.is_recording:
            if not messagebox.askyesno("Exit", "Recording in progress. Stop and exit?"):
                return
            self._stop_recording()
        
        # Cleanup
        if hasattr(self, 'transcription_manager'):
            self.transcription_manager.cleanup()
        
        self.root.destroy()

    def run(self):
        """Start the application"""
        self.root.mainloop()