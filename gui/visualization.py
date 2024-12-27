import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import queue
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class VisualizerConfig:
    """Configuration for the audio visualizer"""
    window_size: int = 4096
    update_interval: int = 50  # milliseconds
    max_frequency: int = 20000  # Hz
    sample_rate: int = 44100
    waveform_color: str = '#2196F3'  # Material Blue
    spectrum_color: str = '#4CAF50'  # Material Green
    background_color: str = '#FAFAFA'
    grid_color: str = '#E0E0E0'

class AudioVisualizer:
    """Real-time audio visualization component"""
    
    def __init__(self, parent: tk.Widget, config: Optional[VisualizerConfig] = None):
        self.parent = parent
        self.config = config or VisualizerConfig()
        self.queue = queue.Queue()
        
        # Create main frame
        self.frame = ttk.LabelFrame(parent, text="Audio Visualization", padding="5")
        
        # Initialize data buffers
        self.waveform_data = np.zeros(self.config.window_size)
        self.spectrum_data = np.zeros(self.config.window_size // 2)
        
        # Setup components
        self._create_plots()
        self._create_controls()
        self._create_info_panel()
        self._start_update_loop()
        
    def _create_plots(self):
        """Create the visualization plots"""
        # Create figure with subplots
        self.fig = Figure(figsize=(8, 4), facecolor=self.config.background_color)
        
        # Waveform subplot
        self.waveform_ax = self.fig.add_subplot(211)
        self.waveform_line, = self.waveform_ax.plot(
            np.arange(self.config.window_size),
            self.waveform_data,
            color=self.config.waveform_color,
            linewidth=1
        )
        self._setup_waveform_axes()
        
        # Spectrum subplot
        self.spectrum_ax = self.fig.add_subplot(212)
        freqs = np.linspace(0, self.config.sample_rate/2, self.config.window_size//2)
        self.spectrum_line, = self.spectrum_ax.plot(
            freqs,
            self.spectrum_data,
            color=self.config.spectrum_color,
            linewidth=1
        )
        self._setup_spectrum_axes()
        
        # Adjust layout and create canvas
        self.fig.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _setup_waveform_axes(self):
        """Configure waveform plot axes"""
        self.waveform_ax.set_title("Waveform", pad=10)
        self.waveform_ax.set_ylim(-1, 1)
        self.waveform_ax.set_xlim(0, self.config.window_size)
        self.waveform_ax.grid(True, color=self.config.grid_color)
        self.waveform_ax.set_facecolor(self.config.background_color)
        
    def _setup_spectrum_axes(self):
        """Configure spectrum plot axes"""
        self.spectrum_ax.set_title("Frequency Spectrum", pad=10)
        self.spectrum_ax.set_ylim(0, 1)
        self.spectrum_ax.set_xlim(0, self.config.max_frequency)
        self.spectrum_ax.grid(True, color=self.config.grid_color)
        self.spectrum_ax.set_facecolor(self.config.background_color)
        
    def _create_controls(self):
        """Create control panel for visualization settings"""
        control_frame = ttk.Frame(self.frame)
        control_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Window size control
        size_frame = ttk.LabelFrame(control_frame, text="Window Size", padding="5")
        size_frame.pack(side=tk.LEFT, padx=5)
        
        self.window_var = tk.StringVar(value=str(self.config.window_size))
        sizes = ['1024', '2048', '4096', '8192']
        window_combo = ttk.Combobox(
            size_frame,
            textvariable=self.window_var,
            values=sizes,
            width=8,
            state="readonly"
        )
        window_combo.pack(side=tk.LEFT)
        window_combo.bind('<<ComboboxSelected>>', self._on_window_size_change)
        
        # View controls
        view_frame = ttk.LabelFrame(control_frame, text="View Options", padding="5")
        view_frame.pack(side=tk.LEFT, padx=5)
        
        # Checkbuttons for each plot
        self.show_waveform = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            view_frame,
            text="Waveform",
            variable=self.show_waveform,
            command=self._update_plot_visibility
        ).pack(side=tk.LEFT, padx=5)
        
        self.show_spectrum = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            view_frame,
            text="Spectrum",
            variable=self.show_spectrum,
            command=self._update_plot_visibility
        ).pack(side=tk.LEFT, padx=5)
        
    def _create_info_panel(self):
        """Create information panel for audio metrics"""
        info_frame = ttk.Frame(self.frame)
        info_frame.pack(fill=tk.X, pady=5)
        
        # Peak level indicator
        self.peak_var = tk.StringVar(value="Peak: 0 dB")
        ttk.Label(info_frame, textvariable=self.peak_var).pack(side=tk.LEFT, padx=10)
        
        # RMS level indicator
        self.rms_var = tk.StringVar(value="RMS: -inf dB")
        ttk.Label(info_frame, textvariable=self.rms_var).pack(side=tk.LEFT, padx=10)
        
    def update(self, audio_data: np.ndarray):
        """Add new audio data to the processing queue"""
        self.queue.put(audio_data)
        
    def _start_update_loop(self):
        """Start the visualization update loop"""
        def update():
            self._process_audio_queue()
            self.frame.after(self.config.update_interval, update)
        update()
        
    def _process_audio_queue(self):
        """Process all queued audio data"""
        try:
            while True:
                audio_data = self.queue.get_nowait()
                self._update_visualizations(audio_data)
        except queue.Empty:
            pass
            
    def _update_visualizations(self, audio_data: np.ndarray):
        """Update both visualizations with new data"""
        # Update waveform buffer
        audio_flat = audio_data.flatten()
        self.waveform_data = np.roll(self.waveform_data, -len(audio_flat))
        self.waveform_data[-len(audio_flat):] = audio_flat
        
        # Update spectrum buffer
        if len(audio_flat) >= 2:
            spectrum = np.abs(np.fft.rfft(audio_flat))
            spectrum = spectrum / len(spectrum)
            self.spectrum_data = np.roll(self.spectrum_data, -len(spectrum))
            self.spectrum_data[-len(spectrum):] = spectrum
        
        # Update plots if visible
        if self.show_waveform.get():
            self.waveform_line.set_ydata(self.waveform_data)
            
        if self.show_spectrum.get():
            self.spectrum_line.set_ydata(self.spectrum_data)
        
        # Update audio metrics
        self._update_audio_metrics(audio_flat)
        
        # Redraw canvas
        self.canvas.draw_idle()
        
    def _update_audio_metrics(self, audio_data: np.ndarray):
        """Update audio level metrics"""
        if len(audio_data) > 0:
            peak = np.max(np.abs(audio_data))
            rms = np.sqrt(np.mean(np.square(audio_data)))
            
            # Convert to dB
            peak_db = 20 * np.log10(peak) if peak > 0 else -np.inf
            rms_db = 20 * np.log10(rms) if rms > 0 else -np.inf
            
            self.peak_var.set(f"Peak: {peak_db:.1f} dB")
            self.rms_var.set(f"RMS: {rms_db:.1f} dB")
            
    def _on_window_size_change(self, event):
        """Handle window size change"""
        new_size = int(self.window_var.get())
        if new_size != self.config.window_size:
            self.config.window_size = new_size
            self.waveform_data = np.zeros(new_size)
            self.spectrum_data = np.zeros(new_size // 2)
            self.waveform_ax.set_xlim(0, new_size)
            self.waveform_line.set_data(np.arange(new_size), self.waveform_data)
            self.canvas.draw()
            
    def _update_plot_visibility(self):
        """Update plot visibility based on checkbutton states"""
        self.waveform_ax.set_visible(self.show_waveform.get())
        self.spectrum_ax.set_visible(self.show_spectrum.get())
        self.fig.tight_layout()
        self.canvas.draw()
    
    def clear(self):
        """Clear all visualization data"""
        self.waveform_data.fill(0)
        self.spectrum_data.fill(0)
        self.waveform_line.set_ydata(self.waveform_data)
        self.spectrum_line.set_ydata(self.spectrum_data)
        self.canvas.draw()