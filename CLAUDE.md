# CLAUDE.md - Project Context & Notes

## Project Overview
**Audio Visualizer** - A retro-style spectrum analyzer that captures system audio and displays it as segmented frequency bars with a rainbow gradient, inspired by classic WinAmp visualizers.

## Current Status
- ✅ Core visualizer implemented with 38-band spectrum analysis
- ✅ Segmented bar rendering with retro glow effects
- ✅ Rainbow gradient color scheme
- ✅ Classic green terminal color scheme
- ✅ Peak hold indicators
- ✅ Interactive control panel with sliders and buttons
- ✅ Real-time bar count adjustment (10-100)
- ✅ Real-time sensitivity control
- ✅ Color scheme toggle (Rainbow ↔ Green)
- ✅ Mouse interaction support
- ✅ Resizable window with responsive controls
- ✅ Git repository initialized and pushed to GitHub
- ✅ GitHub repository: https://github.com/DKP-810/Music_Visualizer

## Recent Changes

### Session 1 (2025-11-04)

#### Part 1: Retro Segmented Bar Design
**Implemented Retro Segmented Bar Design**
- Updated visualization from solid bars to segmented LED-style bars
- Each bar now consists of horizontal segments (4px height, 2px gap)
- Added 3-layer glow effect for retro fuzzy appearance:
  - Outer glow: 30% brightness
  - Middle glow: 60% brightness
  - Bright center: 120% brightness
- Updated color gradient to match reference image: Green → Cyan → Blue → Purple → Pink → Red → Orange → Yellow → Green
- Modified peak indicators to match segmented style with glow

**Key Files Modified:**
- `visualizer.py:79-113` - `_generate_rainbow_gradient()` - Updated color transitions
- `visualizer.py:199-239` - `_draw_bar()` - Complete rewrite for segmented rendering
- `visualizer.py:240-260` - `_draw_peak()` - Updated for glowing peak indicators

#### Part 2: Interactive Control Panel
**Added Real-Time Control Panel**
- Created 200px control dock on right side of visualizer
- Implemented interactive sliders and toggle button
- Added green monochrome color scheme (classic terminal style)
- All controls work in real-time without interrupting playback

**New Features:**
- **Bar Count Slider** - Adjust 10-100 bars with smooth interpolation
- **Sensitivity Slider** - Inverted scale (right = more sensitive)
- **Color Scheme Toggle** - Switch between Rainbow and Green modes
- **Mouse Interaction** - Click, drag, and release support
- **Responsive UI** - Controls reposition on window resize

**Key Files Modified:**
- `visualizer.py:37-72` - Added control panel initialization and UI state
- `visualizer.py:115-127` - `_generate_green_gradient()` - Classic terminal green
- `visualizer.py:136-289` - Control creation, drawing, and interaction handlers
- `visualizer.py:502-533` - Updated draw() to render control panel
- `visualizer.py:569-590` - Added mouse event handling in run()

**Git Configuration:**
- User: DKP-810
- Email: DKP-810@users.noreply.github.com
- Repository: https://github.com/DKP-810/Music_Visualizer

## Technical Architecture

### Audio Capture
- Uses WASAPI loopback to capture system audio (Windows only)
- Sample rate: 48kHz (adapts to device)
- Chunk size: 2048 samples
- Thread-based audio capture to avoid blocking main loop

### Audio Processing
- FFT-based frequency analysis using scipy
- 38 logarithmically-spaced frequency bands (20Hz - 16kHz)
- Hann window for spectral leakage reduction
- dB scaling with configurable sensitivity (min_db=20, max_db=75)
- Smoothing factor: 0.7 for fluid animation

### Visualization
- Pygame-based rendering at 60 FPS
- Segmented bars: 4px segments with 2px gaps
- Multi-layer glow effect for retro aesthetic
- Peak hold time: 30 frames (~0.5 seconds)
- Peak fall speed: 0.5 pixels/frame
- Rainbow gradient: 256-color smooth transitions

## Project Structure
```
Music_Visualizer/
├── visualizer.py          # Main application
├── requirements.txt       # Python dependencies
├── README.md             # User documentation
├── build_exe.py          # PyInstaller build script
├── install.bat           # Windows installation helper
├── sample_visual.png     # Reference image for design
├── .gitignore           # Git ignore rules
└── CLAUDE.md            # This file
```

## Dependencies
- pygame >= 2.5.0 - Graphics and window management
- numpy >= 1.24.0 - Array operations
- pyaudiowpatch >= 0.2.12.7 - WASAPI loopback audio capture
- scipy >= 1.11.0 - FFT and signal processing
- pyinstaller >= 6.0.0 - Executable building

## Customization Points
Users can modify these parameters in `visualizer.py`:

**Visualization:**
- `num_bars` (default: 38) - Number of frequency bands
- `segment_height` (line 210: 4) - Height of each LED segment
- `segment_gap` (line 211: 2) - Gap between segments
- `peak_hold_time` (default: 30 frames) - Peak hold duration
- `peak_fall_speed` (default: 0.5) - Peak fall rate
- `bar_smoothing` (default: 0.7) - Animation smoothness

**Audio Sensitivity:**
- `min_db` (line 273: 20) - Minimum dB threshold
- `max_db` (line 274: 75) - Maximum dB for full height

## Future Enhancement Ideas
- [ ] Add color theme presets (classic green, blue ice, fire, etc.)
- [ ] Implement different visualization modes (waveform, circular, etc.)
- [ ] Add audio recording/export functionality
- [ ] Create settings/config file for easy customization
- [ ] Add beat detection and reactive effects
- [ ] Support multiple monitor/display modes
- [ ] Add keyboard shortcuts for real-time adjustments
- [ ] Implement FPS counter and performance metrics
- [ ] Add fade-in/fade-out transitions
- [ ] Create installer package

## Known Issues & Notes
- Windows only (WASAPI loopback)
- May require administrator privileges on some systems
- Build artifacts (build/, dist/) are gitignored
- First executable run may be slow due to extraction

## Development Commands
```bash
# Run from source
python visualizer.py

# Install dependencies
pip install -r requirements.txt

# Build executable
python build_exe.py

# Git operations
git status
git add .
git commit -m "message"
git push
```

## Reference Materials
- Reference image: `sample_visual.png` - Shows desired segmented retro aesthetic
- Original inspiration: WinAmp classic visualizers

## Notes for Future Sessions
- The segmented bar design was implemented to match the reference image
- Glow effect is achieved through layered rectangles with different opacity/brightness
- Color gradient uses 8 transition zones for smooth rainbow effect
- Peak indicators use white centers with colored glow matching bar position
- All visual parameters are easily adjustable via constants in the code
