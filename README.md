# Audio Visualizer - Classic WinAmp Style

A nostalgic audio visualizer that captures your system audio and displays a beautiful 38-band spectrum analyzer with rainbow gradient colors and peak hold effects, just like the classic WinAmp visualizations!

## Features

- **38 frequency bands** for detailed audio analysis
- **Rainbow gradient** colors (green → cyan → blue → purple → red → yellow)
- **Peak hold effect** - peaks pause at the highest point before falling
- **Resizable window** with 16:9 default aspect ratio (1280x720)
- **Real-time system audio capture** - visualizes whatever is playing on your PC

## Installation

### Option 1: Run from Python

1. Install Python 3.8 or higher
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the visualizer:
```bash
python visualizer.py
```

### Option 2: Run as Executable (After Building)

Simply double-click `AudioVisualizer.exe` - no Python installation required!

## Important: Windows Audio Setup

This visualizer uses **WASAPI loopback** to capture system audio directly - no special setup required! It will automatically detect and capture audio from your default playback device.

**Just make sure:**
- Audio is actually playing on your system
- Your default playback device is set correctly in Windows Sound settings

**Note:** If you encounter any issues:
- Try running the application as administrator
- Make sure no other application is exclusively using the audio device
- Check Windows Sound settings to ensure your speakers/headphones are the default device

## Building an Executable

To create a standalone .exe file:

```bash
python build_exe.py
```

The executable will be created in the `dist` folder.

## Controls

- **ESC** or close window to exit
- Resize the window by dragging the edges

## Customization

You can modify these parameters in `visualizer.py`:

- `num_bars` - Number of frequency bars (default: 38)
- `peak_hold_time` - How long peaks stay at the top (default: 30 frames)
- `peak_fall_speed` - How fast peaks fall (default: 0.5)
- `bar_smoothing` - Smoothness of bar animation (default: 0.7)
- `min_db` / `max_db` - Sensitivity adjustment

## Troubleshooting

**No bars moving:**
- Make sure audio is playing
- Check that Stereo Mix is enabled
- Try running as administrator
- Check the console for error messages

**Bars too sensitive/not sensitive enough:**
- Adjust `min_db` and `max_db` values in the code
- Default is 40-100, try 30-90 for more sensitivity or 50-110 for less

## License

Free to use and modify!
