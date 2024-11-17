from gui.recorder_gui import AudioRecorderGUI

def main():
    app = AudioRecorderGUI()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
        if hasattr(app, 'stream') and app.stream is not None:
            app.stop_recording()

if __name__ == "__main__":
    main()
