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
- ✅ Glass reflection effect with 45-degree angle
- ✅ Adjustable reflection intensity (0-100%)
- ✅ Gaussian blur effect for reflections (0-10 radius)
- ✅ Reflection toggle button with conditional sliders
- ✅ Mouse interaction support
- ✅ Resizable window with responsive controls
- ✅ Git repository initialized and pushed to GitHub
- ✅ GitHub repository: https://github.com/DKP-810/Music_Visualizer
- ✅ Feature branch: `reflection-enabled`

## Recent Changes

### Session 2 (2025-11-04) - 45-Degree Reflection & Water Ripple Effect

#### Part 1: 45-Degree Angled Reflection
**Implemented Angled Glass Reflection**
- Changed reflection from vertical mirror to 45-degree diagonal
- Reflections now shift horizontally as they extend downward
- Formula: `angle_offset_x = distance_from_glass` for perfect 45° angle
- Added bounds checking to prevent off-screen rendering
- Both bar segments and peak indicators use angled reflection

**Key Files Modified:**
- `visualizer.py:303-320` - Added 45-degree offset calculation in `_draw_bar_reflection()`
- `visualizer.py:358-373` - Added matching offset in `_draw_peak_reflection()`

**User Feedback:** "It worked great!"

#### Part 2: Gaussian Blur Effect for Reflections
**Replaced Water Ripple with Gaussian-Style Blur**
- Removed water ripple distortion feature (didn't achieve desired effect)
- Implemented adjustable blur effect applied only to reflections
- New "Blur Amount" slider (0-10) controlling blur intensity
- Default: 2 (subtle blur for glass effect)
- Conditional UI: slider only visible when reflection enabled

**Technical Implementation:**
- Uses `reflection_blur` parameter (0-10 pixels radius)
- Box blur approximation using pygame smoothscale downsampling/upsampling
- Reflections rendered to temporary SRCALPHA surface
- Blur applied via iterative 50% downscale → upscale cycles
- Blurred surface composited onto main screen at glass_y position
- More iterations = stronger blur effect

**Key Files Modified:**
- `visualizer.py:61` - Changed `reflection_roughness` to `reflection_blur` parameter (default 2)
- `visualizer.py:219-232` - Updated slider control to "Blur Amount" (0-10 range)
- `visualizer.py:320-322` - Updated slider handler for blur radius
- `visualizer.py:335-353` - Added `_apply_blur()` helper method
- `visualizer.py:748-781` - Modified reflection rendering to use temporary surface with blur
- `visualizer.py:565-570` - Removed ripple X-offset distortion from bar reflection
- `visualizer.py:610-614` - Removed ripple X-offset distortion from peak reflection

**Development Process:**
- Initially implemented water ripple with X+Y offsets (Part 2a)
- User reported physics issues, revised to X-offset only (Part 2b)
- User decided ripple effect didn't look as desired
- Replaced with blur effect for softer, more realistic glass/water appearance

**User Feedback:** User requested removal of ripple feature in favor of blur

#### Part 3: App-Specific Audio Capture Research
**Research Only - Not Implemented**
User asked about isolating audio by application (e.g., YouTube Music only). Provided analysis of options:
1. WASAPI Per-Application Capture (limited support)
2. Virtual Audio Cable (requires extra software)
3. App Volume API with pycaw (best option - enumerate audio sessions)
4. Process Injection (complex/risky)

**Decision:** User chose not to pursue this feature yet

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

#### Part 3: Glass Reflection Effect
**Implemented Glass Reflection with Adjustable Intensity**
- Added glass surface line at 60% down the screen
- Bars grow upward from glass surface
- Mirrored reflections extend downward with exponential fade
- Desaturated reflection colors (85% brightness) for realism
- Adjustable reflection intensity (0-100%)
- Conditional UI: intensity slider only visible when reflection enabled

**New Features:**
- **Reflection Toggle** - Turn glass reflection effect on/off
- **Reflection Intensity Slider** - Adjust opacity (0-100%, default 60%)
- **Exponential Fade** - fade_ratio^1.5 for realistic gradient
- **Peak Reflections** - Use half intensity of bar reflections
- **Split Layout** - 60% bars / 40% reflection area

**Key Files Modified:**
- `visualizer.py:55-60` - Added reflection_enabled and reflection_intensity parameters
- `visualizer.py:186-217` - Added reflection toggle button and intensity slider
- `visualizer.py:256-300` - Updated _draw_bar() with base_y anchor parameter
- `visualizer.py:507-598` - New _draw_bar_reflection() and _draw_peak_reflection() methods
- `visualizer.py:653-706` - Updated draw() with glass surface layout and reflection rendering
- `visualizer.py:713-717` - Added conditional control visibility logic

**Bug Fixes:**
- Fixed bar drawing anchor point from self.height to glass_y for proper mirroring
- Fixed sensitivity slider inversion (right = more sensitive)
- Fixed reflection darkness with adjustable intensity control

**Git Configuration:**
- User: DKP-810
- Email: DKP-810@users.noreply.github.com
- Repository: https://github.com/DKP-810/Music_Visualizer
- Main branch: `master`
- Feature branch: `reflection-enabled`

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
- Glass reflection: 60/40 split layout with exponential fade
- 45-degree angled reflection (diagonal distortion)
- Reflection intensity: Adjustable 0.0-1.0 (default 0.6)
- Reflection blur: Box blur approximation (0-10 radius, default 2)

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
- `segment_height` (line 213: 4) - Height of each LED segment
- `segment_gap` (line 214: 2) - Gap between segments
- `peak_hold_time` (default: 30 frames) - Peak hold duration
- `peak_fall_speed` (default: 0.5) - Peak fall rate
- `bar_smoothing` (default: 0.7) - Animation smoothness
- `reflection_enabled` (line 59: True) - Enable glass reflection
- `reflection_intensity` (line 60: 0.6) - Reflection opacity (0.0-1.0)
- `reflection_blur` (line 61: 2) - Reflection blur radius (0-10 pixels)

**Audio Sensitivity:**
- `min_db` (line 56: 20) - Minimum dB threshold
- `max_db` (line 57: 75) - Maximum dB for full height

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

# Branch operations
git branch                    # List branches
git checkout branch-name      # Switch to branch
git checkout -b new-branch    # Create and switch to new branch
git merge branch-name         # Merge branch into current branch
```

## Git Branching Strategy
**Current Branches:**
- `master` - Main development branch (includes all features)
- `reflection-enabled` - Feature branch created from master at reflection commit

**Note:** Both branches currently have identical code. The `reflection-enabled` branch was created to demonstrate branching workflow. In the future, you could:
- Create a `no-reflection` branch from an earlier commit (before reflection was added)
- Keep master as the full-featured version
- Maintain different versions for different use cases

## Reference Materials
- Reference image: `sample_visual.png` - Shows desired segmented retro aesthetic
- Original inspiration: WinAmp classic visualizers

## Notes for Future Sessions
- The segmented bar design was implemented to match the reference image
- Glow effect is achieved through layered rectangles with different opacity/brightness
- Color gradient uses 8 transition zones for smooth rainbow effect
- Peak indicators use white centers with colored glow matching bar position
- All visual parameters are easily adjustable via constants in the code
