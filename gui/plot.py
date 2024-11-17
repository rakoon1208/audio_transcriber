import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AudioPlot:
    def __init__(self, root):
        # Figure and axes for plotting
        self.fig, self.ax = plt.subplots(figsize=(8, 2))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        self.buffer_size = 16000  
        self.plot_data = np.zeros(self.buffer_size, dtype=np.float32)  
        self.line, = self.ax.plot(np.arange(self.buffer_size), self.plot_data, color='blue')

        self.ax.set_ylim(-2, 2)  
        self.ax.set_title("Audio Level")
        self.ax.set_xlabel("Samples")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)

    def update_plot(self, audio_recorder):
        data = audio_recorder.get_audio_data()
        if data is not None:
            # Apply a higher scaling factor to amplify the visual amplitude
            scaling_factor = 1.5  # Adjust this to a value greater than 1 to increase the amplitude
            scaled_data = data * scaling_factor
            
            update_size = len(scaled_data)
            self.plot_data = np.roll(self.plot_data, -update_size)
            self.plot_data[-update_size:] = scaled_data
            self.line.set_ydata(self.plot_data)
            self.canvas.draw()
