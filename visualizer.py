"""
Audio Visualizer - Classic WinAmp Style Spectrum Analyzer
Captures system audio and displays a 38-band frequency spectrum with rainbow gradient
"""

import pygame
import numpy as np
import pyaudiowpatch as pyaudio
from scipy import signal
import threading
import queue
import sys
import struct
import ctypes
import os
import random

# Windows DPI awareness fix - must be called before pygame.init()
if sys.platform == 'win32':
    try:
        # Tell Windows we're DPI aware so it doesn't scale our window
        ctypes.windll.user32.SetProcessDPIAware()
    except:
        pass

class Star:
    """Represents a single star in the starfield"""
    def __init__(self, x, y, size, speed, color, layer):
        self.x = x
        self.y = y
        self.size = size  # Star size (0.5 to 3.0)
        self.speed = speed  # Horizontal scroll speed (base speed)
        self.color = color  # RGB tuple
        self.layer = layer  # 0=back, 1=mid, 2=front
        self.brightness = random.uniform(0.3, 1.0)  # For twinkling effect
        self.twinkle_speed = random.uniform(0.01, 0.05)
        self.twinkle_direction = 1
        self.trail_positions = []  # List of (x, y) positions for warp streak effect

class AudioVisualizer:
    def __init__(self, width=1280, height=720, num_bars=38):
        pygame.init()

        # Window settings - store the REQUESTED dimensions before any scaling
        self.original_width = width
        self.original_height = height
        self.width = width
        self.height = height
        self.num_bars = num_bars

        # Get display info for better default sizing
        display_info = pygame.display.Info()
        screen_width = display_info.current_w
        screen_height = display_info.current_h

        # Start with 80% of screen size (looks good on any monitor)
        default_width = int(screen_width * 0.8)
        default_height = int(screen_height * 0.6)

        # Use provided dimensions or defaults
        self.width = default_width if width == 1280 else width
        self.height = default_height if height == 720 else height

        # Update original dimensions if we're using defaults
        self.original_width = self.width
        self.original_height = self.height

        # Menu bar settings (replaces side panel)
        self.menu_height = 35
        self.menu_visible = True
        self.last_mouse_move_time = pygame.time.get_ticks()
        self.menu_hide_delay = 3000  # Hide after 3 seconds of inactivity
        self.menu_show_threshold = 50  # Show when mouse is within 50px of top
        self.open_dropdown = None  # Track which dropdown menu is open

        # Visualizer now uses full width
        self.visualizer_width = self.width

        # Create resizable windowed mode (explicitly NOT fullscreen)
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("Audio Visualizer")

        # Fullscreen state
        self.is_fullscreen = False

        # Font for UI
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 12)
        self.font_small = pygame.font.SysFont('Arial', 10)

        # Audio settings
        self.sample_rate = 48000
        self.chunk_size = 2048
        self.audio_queue = queue.Queue()

        # Visualization parameters (now adjustable)
        self.min_db = 20
        self.max_db = 75
        self.color_scheme = "rainbow"  # "rainbow" or "green"
        self.reflection_enabled = True  # Glass reflection effect
        self.reflection_intensity = 0.6  # Reflection opacity multiplier (0.0 - 1.0)
        self.reflection_blur = 2  # Blur radius for reflection (0 - 10 pixels)
        self.starfield_enabled = True  # Animated starfield background
        self.starfield_speed = 1.0  # Speed multiplier for starfield (0.0 - 3.0)

        # Visualization data
        self.bar_heights = np.zeros(num_bars)
        self.peak_heights = np.zeros(num_bars)
        self.peak_hold_time = np.zeros(num_bars)
        self.peak_fall_speed = 0.5  # Speed at which peaks fall
        self.bar_smoothing = 0.7  # Smoothing factor for bar movements

        # Colors - Rainbow gradient
        self.colors = self._generate_rainbow_gradient()

        # UI Controls - organized into dropdown menus
        self.menu_items = self._create_menu_structure()
        self.dragging_control = None

        # Starfield
        self.stars = []
        self._initialize_starfield()

        # Audio capture thread
        self.running = True
        self.audio_thread = threading.Thread(target=self._capture_audio, daemon=True)
        self.audio_thread.start()

    def _generate_rainbow_gradient(self):
        """Generate rainbow color gradient matching the reference image"""
        colors = []
        steps = 256

        # Create rainbow: Green -> Cyan -> Blue -> Purple -> Pink -> Red -> Orange -> Yellow -> Green
        for i in range(steps):
            t = i / steps

            if t < 1/8:  # Green to Cyan
                ratio = t / (1/8)
                colors.append((0, 255, int(255 * ratio)))
            elif t < 2/8:  # Cyan to Blue
                ratio = (t - 1/8) / (1/8)
                colors.append((0, int(255 * (1-ratio)), 255))
            elif t < 3/8:  # Blue to Purple
                ratio = (t - 2/8) / (1/8)
                colors.append((int(128 * ratio), 0, 255))
            elif t < 4/8:  # Purple to Pink/Magenta
                ratio = (t - 3/8) / (1/8)
                colors.append((int(128 + 127 * ratio), 0, 255))
            elif t < 5/8:  # Pink to Red
                ratio = (t - 4/8) / (1/8)
                colors.append((255, 0, int(255 * (1-ratio))))
            elif t < 6/8:  # Red to Orange
                ratio = (t - 5/8) / (1/8)
                colors.append((255, int(128 * ratio), 0))
            elif t < 7/8:  # Orange to Yellow
                ratio = (t - 6/8) / (1/8)
                colors.append((255, int(128 + 127 * ratio), 0))
            else:  # Yellow to Green
                ratio = (t - 7/8) / (1/8)
                colors.append((int(255 * (1-ratio)), 255, 0))

        return colors

    def _generate_green_gradient(self):
        """Generate classic green monochrome terminal color scheme"""
        colors = []
        steps = 256

        # Classic terminal green with slight variations
        for i in range(steps):
            # Gradient from dark green to bright green
            intensity = i / steps
            green_value = int(80 + 175 * intensity)  # Range from dark to bright green
            colors.append((0, green_value, int(green_value * 0.3)))  # Slight yellow tint for authenticity

        return colors

    def _update_colors(self):
        """Update color scheme based on current setting"""
        if self.color_scheme == "rainbow":
            self.colors = self._generate_rainbow_gradient()
        else:  # green
            self.colors = self._generate_green_gradient()

    def _initialize_starfield(self):
        """Initialize the 3-layer parallax starfield"""
        self.stars = []

        # Star colors - mix of white, yellow, purple, cyan like in the reference
        star_colors = [
            (255, 255, 255),  # White
            (255, 255, 200),  # Pale yellow
            (200, 200, 255),  # Pale blue
            (255, 200, 255),  # Pale magenta/purple
            (200, 255, 255),  # Pale cyan
            (255, 220, 180),  # Warm white
        ]

        # Layer 0: Background - small, slow, dim stars
        for _ in range(30):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            size = random.uniform(0.5, 1.0)
            speed = random.uniform(0.1, 0.3)
            color = random.choice(star_colors)
            self.stars.append(Star(x, y, size, speed, color, 0))

        # Layer 1: Midground - medium stars
        for _ in range(20):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            size = random.uniform(1.0, 2.0)
            speed = random.uniform(0.3, 0.6)
            color = random.choice(star_colors)
            self.stars.append(Star(x, y, size, speed, color, 1))

        # Layer 2: Foreground - large, fast, bright stars
        for _ in range(10):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            size = random.uniform(2.0, 3.5)
            speed = random.uniform(0.6, 1.2)
            color = random.choice(star_colors)
            self.stars.append(Star(x, y, size, speed, color, 2))

    def _create_menu_structure(self):
        """Create dropdown menu structure with organized controls"""
        menu_items = []

        # Calculate menu item positions (horizontal layout)
        menu_item_width = 80
        menu_item_spacing = 10
        menu_start_x = 10

        # Audio Menu
        audio_controls = [
            {
                'type': 'slider',
                'name': 'Bars',
                'width': 180,
                'height': 20,
                'min': 10,
                'max': 100,
                'value': self.num_bars,
                'param': 'num_bars'
            },
            {
                'type': 'slider',
                'name': 'Sensitivity',
                'width': 180,
                'height': 20,
                'min': 10,
                'max': 50,
                'value': 60 - self.min_db,
                'param': 'min_db'
            }
        ]

        menu_items.append({
            'name': 'Audio',
            'x': menu_start_x,
            'y': 0,
            'width': menu_item_width,
            'height': self.menu_height,
            'controls': audio_controls
        })

        # Visual Menu
        visual_controls = [
            {
                'type': 'button',
                'name': 'Color Scheme',
                'width': 180,
                'height': 35,
                'param': 'color_scheme',
                'states': ['rainbow', 'green'],
                'state_index': 0
            },
            {
                'type': 'button',
                'name': 'Starfield',
                'width': 180,
                'height': 35,
                'param': 'starfield',
                'states': ['on', 'off'],
                'state_index': 0
            },
            {
                'type': 'slider',
                'name': 'Starfield Speed',
                'width': 180,
                'height': 20,
                'min': 0,
                'max': 300,
                'value': int(self.starfield_speed * 100),
                'param': 'starfield_speed',
                'visible_when': 'starfield_enabled'
            },
            {
                'type': 'button',
                'name': 'Reflection',
                'width': 180,
                'height': 35,
                'param': 'reflection',
                'states': ['on', 'off'],
                'state_index': 0
            },
            {
                'type': 'slider',
                'name': 'Reflection Intensity',
                'width': 180,
                'height': 20,
                'min': 0,
                'max': 100,
                'value': int(self.reflection_intensity * 100),
                'param': 'reflection_intensity',
                'visible_when': 'reflection_enabled'
            },
            {
                'type': 'slider',
                'name': 'Blur Amount',
                'width': 180,
                'height': 20,
                'min': 0,
                'max': 10,
                'value': self.reflection_blur,
                'param': 'reflection_blur',
                'visible_when': 'reflection_enabled'
            }
        ]

        menu_items.append({
            'name': 'Visual',
            'x': menu_start_x + menu_item_width + menu_item_spacing,
            'y': 0,
            'width': menu_item_width,
            'height': self.menu_height,
            'controls': visual_controls
        })

        # Animation Menu
        animation_controls = [
            {
                'type': 'slider',
                'name': 'Peak Fall Speed',
                'width': 180,
                'height': 20,
                'min': 1,
                'max': 50,
                'value': int(self.peak_fall_speed * 10),
                'param': 'peak_fall_speed'
            }
        ]

        menu_items.append({
            'name': 'Animation',
            'x': menu_start_x + (menu_item_width + menu_item_spacing) * 2,
            'y': 0,
            'width': menu_item_width,
            'height': self.menu_height,
            'controls': animation_controls
        })

        return menu_items

    def _draw_slider(self, control):
        """Draw a slider control"""
        x, y, width, height = control['x'], control['y'], control['width'], control['height']

        # Label
        label_surface = self.font.render(control['name'], True, (200, 200, 200))
        self.screen.blit(label_surface, (x, y - 18))

        # Track
        pygame.draw.rect(self.screen, (60, 60, 60), (x, y, width, height))
        pygame.draw.rect(self.screen, (100, 100, 100), (x, y, width, height), 1)

        # Handle position
        value_range = control['max'] - control['min']
        normalized = (control['value'] - control['min']) / value_range
        handle_x = x + int(normalized * width)

        # Filled portion
        pygame.draw.rect(self.screen, (0, 150, 255), (x, y, handle_x - x, height))

        # Handle
        handle_width = 10
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (handle_x - handle_width // 2, y - 2, handle_width, height + 4))
        pygame.draw.rect(self.screen, (200, 200, 200),
                        (handle_x - handle_width // 2, y - 2, handle_width, height + 4), 1)

        # Value display
        value_text = self.font_small.render(str(int(control['value'])), True, (200, 200, 200))
        self.screen.blit(value_text, (x + width + 5, y + 4))

    def _draw_button(self, control):
        """Draw a button control"""
        x, y, width, height = control['x'], control['y'], control['width'], control['height']

        # Label
        label_surface = self.font.render(control['name'], True, (200, 200, 200))
        self.screen.blit(label_surface, (x, y - 18))

        # Button background - color based on parameter type
        current_state = control['states'][control['state_index']]

        if control['param'] == 'color_scheme':
            if current_state == 'rainbow':
                color = (100, 50, 150)
            else:
                color = (0, 100, 50)
        elif control['param'] == 'starfield':
            if current_state == 'on':
                color = (80, 60, 120)  # Purple for starfield on
            else:
                color = (60, 60, 60)  # Gray for starfield off
        elif control['param'] == 'reflection':
            if current_state == 'on':
                color = (50, 120, 180)  # Blue for reflection on
            else:
                color = (60, 60, 60)  # Gray for reflection off
        else:
            color = (80, 80, 80)  # Default gray

        pygame.draw.rect(self.screen, color, (x, y, width, height))
        pygame.draw.rect(self.screen, (200, 200, 200), (x, y, width, height), 2)

        # Button text
        text = current_state.upper()
        text_surface = self.font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
        self.screen.blit(text_surface, text_rect)

    def _handle_slider_drag(self, control, mouse_x):
        """Handle dragging a slider"""
        rel_x = mouse_x - control['x']
        rel_x = max(0, min(control['width'], rel_x))

        normalized = rel_x / control['width']
        value_range = control['max'] - control['min']
        new_value = control['min'] + normalized * value_range

        control['value'] = int(new_value)

        # Update the corresponding parameter
        if control['param'] == 'num_bars':
            self._update_bar_count(int(new_value))
        elif control['param'] == 'min_db':
            # Invert sensitivity: higher slider value = lower min_db = more sensitive
            self.min_db = 60 - int(new_value)
        elif control['param'] == 'reflection_intensity':
            # Convert 0-100 to 0.0-1.0
            self.reflection_intensity = int(new_value) / 100.0
        elif control['param'] == 'reflection_blur':
            # Blur radius 0-10 pixels
            self.reflection_blur = int(new_value)
        elif control['param'] == 'peak_fall_speed':
            # Convert slider value (1-50) to fall speed (0.1-5.0)
            self.peak_fall_speed = int(new_value) / 10.0
        elif control['param'] == 'starfield_speed':
            # Convert slider value (0-300) to speed multiplier (0.0-3.0)
            self.starfield_speed = int(new_value) / 100.0

    def _handle_button_click(self, control):
        """Handle clicking a button"""
        # Toggle between states
        control['state_index'] = (control['state_index'] + 1) % len(control['states'])

        if control['param'] == 'color_scheme':
            self.color_scheme = control['states'][control['state_index']]
            self._update_colors()
        elif control['param'] == 'starfield':
            self.starfield_enabled = control['states'][control['state_index']] == 'on'
        elif control['param'] == 'reflection':
            self.reflection_enabled = control['states'][control['state_index']] == 'on'

    def _apply_blur(self, surface, blur_radius):
        """Apply a simple box blur to a surface using downscale/upscale"""
        if blur_radius <= 0:
            return surface

        # Get surface dimensions
        width, height = surface.get_size()

        # Apply blur by scaling down and back up multiple times
        # Each iteration creates a smoother blur
        temp_surface = surface.copy()
        for _ in range(blur_radius):
            # Scale down by 50% and back up - this creates blur effect
            small_w = max(1, width // 2)
            small_h = max(1, height // 2)
            temp_surface = pygame.transform.smoothscale(temp_surface, (small_w, small_h))
            temp_surface = pygame.transform.smoothscale(temp_surface, (width, height))

        return temp_surface

    def _update_bar_count(self, new_count):
        """Update the number of bars dynamically"""
        old_count = self.num_bars
        self.num_bars = new_count

        # Resize arrays
        new_bar_heights = np.zeros(new_count)
        new_peak_heights = np.zeros(new_count)
        new_peak_hold_time = np.zeros(new_count)

        # Copy over existing data (interpolate if needed)
        if old_count > 0:
            for i in range(new_count):
                old_index = int(i * old_count / new_count)
                old_index = min(old_index, old_count - 1)
                new_bar_heights[i] = self.bar_heights[old_index]
                new_peak_heights[i] = self.peak_heights[old_index]
                new_peak_hold_time[i] = self.peak_hold_time[old_index]

        self.bar_heights = new_bar_heights
        self.peak_heights = new_peak_heights
        self.peak_hold_time = new_peak_hold_time

    def _capture_audio(self):
        """Capture system audio in a separate thread"""
        try:
            # Initialize PyAudio
            p = pyaudio.PyAudio()

            # Get default WASAPI loopback device
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

            if not default_speakers["isLoopbackDevice"]:
                # Try to find loopback device
                for loopback in p.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        default_speakers = loopback
                        break

            print(f"Recording from: {default_speakers['name']}")

            # Open audio stream
            stream = p.open(
                format=pyaudio.paInt16,
                channels=default_speakers["maxInputChannels"],
                rate=int(default_speakers["defaultSampleRate"]),
                frames_per_buffer=self.chunk_size,
                input=True,
                input_device_index=default_speakers["index"]
            )

            # Update sample rate to match device
            self.sample_rate = int(default_speakers["defaultSampleRate"])

            print(f"Capture started at {self.sample_rate} Hz")

            while self.running:
                try:
                    # Read audio data
                    data = stream.read(self.chunk_size, exception_on_overflow=False)

                    # Convert bytes to numpy array
                    audio_data = np.frombuffer(data, dtype=np.int16)

                    # Convert to mono if stereo
                    if default_speakers["maxInputChannels"] == 2:
                        audio_data = audio_data.reshape(-1, 2).mean(axis=1)

                    # Normalize to float
                    audio_data = audio_data.astype(np.float32) / 32768.0

                    self.audio_queue.put(audio_data)

                except Exception as e:
                    if self.running:  # Only print if we're still running
                        print(f"Error reading audio: {e}")

            stream.stop_stream()
            stream.close()
            p.terminate()

        except Exception as e:
            print(f"Error capturing audio: {e}")
            print("Make sure audio is playing on your system.")
            print("If the issue persists, try running as administrator.")

    def _process_audio(self, audio_data):
        """Process audio data using FFT and return frequency band amplitudes"""
        # Apply window function to reduce spectral leakage
        windowed = audio_data * signal.windows.hann(len(audio_data))

        # Perform FFT
        fft = np.fft.rfft(windowed)
        magnitude = np.abs(fft)

        # Convert to decibels
        magnitude = 20 * np.log10(magnitude + 1e-10)

        # Group frequencies into bands (logarithmic spacing for better visualization)
        freq_bins = len(magnitude)

        # Create logarithmically spaced frequency bands
        # Focus on frequencies humans can hear well (20Hz - 16kHz)
        max_freq = min(16000, self.sample_rate / 2)
        freq_ranges = np.logspace(np.log10(20), np.log10(max_freq), self.num_bars + 1)

        band_amplitudes = []
        for i in range(self.num_bars):
            low_freq = freq_ranges[i]
            high_freq = freq_ranges[i + 1]

            # Convert frequencies to bin indices
            low_bin = int(low_freq * freq_bins / (self.sample_rate / 2))
            high_bin = int(high_freq * freq_bins / (self.sample_rate / 2))

            # Average the magnitude in this frequency range
            if high_bin > low_bin:
                band_avg = np.mean(magnitude[low_bin:high_bin])
            else:
                band_avg = magnitude[low_bin] if low_bin < len(magnitude) else 0

            band_amplitudes.append(band_avg)

        return np.array(band_amplitudes)

    def _draw_bar(self, x, y, width, height, bar_index, base_y=None):
        """Draw a segmented frequency bar with retro glow effect"""
        if height < 1:
            return

        # If base_y not provided, use the bottom of bar's drawing area
        if base_y is None:
            base_y = y + height

        # Calculate color based on bar position (left to right)
        color_pos = int((bar_index / (self.num_bars - 1)) * (len(self.colors) - 1))
        base_color = self.colors[color_pos]

        # Segment settings
        segment_height = 4  # Height of each segment
        segment_gap = 2     # Gap between segments

        # Calculate how many segments to draw based on bar height
        total_segment_height = segment_height + segment_gap
        num_segments = int(height / total_segment_height)

        # Draw segments from bottom to top
        for seg in range(num_segments):
            seg_y = base_y - ((seg + 1) * total_segment_height)

            if seg_y < y:  # Don't draw above the bar height
                break

            # Create glow effect with multiple layers
            # Outer glow (darkest/largest)
            glow_color_outer = tuple(int(c * 0.3) for c in base_color)
            pygame.draw.rect(self.screen, glow_color_outer,
                           (x - 1, seg_y - 1, width + 2, segment_height + 2))

            # Middle glow
            glow_color_middle = tuple(int(c * 0.6) for c in base_color)
            pygame.draw.rect(self.screen, glow_color_middle,
                           (x, seg_y, width, segment_height))

            # Bright center
            bright_color = tuple(min(255, int(c * 1.2)) for c in base_color)
            center_height = max(1, segment_height - 2)
            pygame.draw.rect(self.screen, bright_color,
                           (x + 1, seg_y + 1, max(1, width - 2), center_height))

    def _draw_peak(self, x, y, width, bar_index):
        """Draw the peak indicator for a bar with glow effect"""
        if y < 0 or y > self.height:
            return

        # Calculate color based on bar position
        color_pos = int((bar_index / (self.num_bars - 1)) * (len(self.colors) - 1))
        base_color = self.colors[color_pos]

        # Peak is a glowing white/bright segment
        peak_height = 4

        # Outer glow
        glow_color = tuple(int(c * 0.5) for c in base_color)
        pygame.draw.rect(self.screen, glow_color,
                        (x - 1, y - 1, width + 2, peak_height + 2))

        # Bright white center
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (x, y, width, peak_height))

    def _draw_bar_reflection(self, x, mirror_y, width, height, bar_index, glass_y):
        """Draw a mirrored reflection of a bar with fade gradient, 45-degree angle, and water ripple distortion"""
        if height < 1:
            return

        # Calculate color based on bar position
        color_pos = int((bar_index / (self.num_bars - 1)) * (len(self.colors) - 1))
        base_color = self.colors[color_pos]

        # Desaturate the reflection color slightly
        desaturated_color = tuple(int(c * 0.85) for c in base_color)

        # Segment settings (same as main bars)
        segment_height = 4
        segment_gap = 2
        total_segment_height = segment_height + segment_gap
        num_segments = int(height / total_segment_height)

        # Draw segments from top to bottom (mirrored) with 45-degree angle
        for seg in range(num_segments):
            seg_y = mirror_y + (seg * total_segment_height)

            # Calculate fade based on distance from glass surface
            distance_from_glass = seg_y - glass_y
            max_reflection_height = self.height - glass_y
            fade_ratio = 1.0 - (distance_from_glass / max_reflection_height)
            fade_ratio = max(0, min(1, fade_ratio))  # Clamp 0-1

            # Apply exponential fade for more realistic look
            fade_ratio = fade_ratio ** 1.5

            # Base opacity for reflection (adjustable via reflection_intensity)
            base_opacity = self.reflection_intensity
            opacity = base_opacity * fade_ratio

            if opacity < 0.01:  # Don't draw nearly invisible segments
                continue

            # Calculate 45-degree angle offset (shifts right as it goes down)
            # For 45 degrees: horizontal offset = vertical offset
            angle_offset_x = distance_from_glass

            # Check if reflection would be off-screen
            seg_x = x + angle_offset_x
            if seg_x > self.visualizer_width or seg_x < -width:
                continue

            # Apply opacity to colors
            glow_outer = tuple(int(c * 0.3 * opacity) for c in desaturated_color)
            glow_middle = tuple(int(c * 0.6 * opacity) for c in desaturated_color)
            bright_center = tuple(int(min(255, c * 1.2) * opacity) for c in desaturated_color)

            # Create surfaces with alpha for proper blending
            # Outer glow
            if opacity > 0.05:
                surf_outer = pygame.Surface((int(width) + 2, segment_height + 2), pygame.SRCALPHA)
                surf_outer.fill((*glow_outer, int(255 * opacity)))
                self.screen.blit(surf_outer, (seg_x - 1, seg_y - 1))

            # Middle glow
            surf_middle = pygame.Surface((int(width), segment_height), pygame.SRCALPHA)
            surf_middle.fill((*glow_middle, int(255 * opacity)))
            self.screen.blit(surf_middle, (seg_x, seg_y))

            # Bright center
            center_h = max(1, segment_height - 2)
            surf_center = pygame.Surface((max(1, int(width) - 2), center_h), pygame.SRCALPHA)
            surf_center.fill((*bright_center, int(255 * opacity * 0.8)))
            self.screen.blit(surf_center, (seg_x + 1, seg_y + 1))

    def _draw_peak_reflection(self, x, mirror_y, width, bar_index, glass_y):
        """Draw a mirrored reflection of a peak indicator with fade, 45-degree angle, and water ripple"""
        # Calculate opacity based on distance from glass
        distance_from_glass = mirror_y - glass_y
        max_reflection_height = self.height - glass_y
        fade_ratio = 1.0 - (distance_from_glass / max_reflection_height)
        fade_ratio = max(0, min(1, fade_ratio)) ** 1.5
        # Peaks use half the reflection intensity
        opacity = (self.reflection_intensity * 0.5) * fade_ratio

        if opacity < 0.01:
            return

        # Calculate 45-degree angle offset
        angle_offset_x = distance_from_glass

        # Check if reflection would be off-screen
        peak_x = x + angle_offset_x
        if peak_x > self.visualizer_width or peak_x < -width:
            return

        # Calculate color
        color_pos = int((bar_index / (self.num_bars - 1)) * (len(self.colors) - 1))
        base_color = self.colors[color_pos]
        desaturated = tuple(int(c * 0.85) for c in base_color)

        peak_height = 4

        # Outer glow
        surf_glow = pygame.Surface((int(width) + 2, peak_height + 2), pygame.SRCALPHA)
        glow_color = tuple(int(c * 0.5 * opacity) for c in desaturated)
        surf_glow.fill((*glow_color, int(255 * opacity)))
        self.screen.blit(surf_glow, (peak_x - 1, mirror_y - 1))

        # White center (dimmed for reflection)
        surf_center = pygame.Surface((int(width), peak_height), pygame.SRCALPHA)
        surf_center.fill((255, 255, 255, int(200 * opacity)))
        self.screen.blit(surf_center, (peak_x, mirror_y))

    def update(self):
        """Update visualization with latest audio data"""
        # Update menu visibility based on mouse position
        self._update_menu_visibility()

        # Update starfield
        if self.starfield_enabled:
            self._update_starfield()

        # Process all available audio chunks
        while not self.audio_queue.empty():
            try:
                audio_data = self.audio_queue.get_nowait()

                # Process audio to get frequency band amplitudes
                amplitudes = self._process_audio(audio_data)

                # Normalize amplitudes to screen height (with some scaling)
                # Use adjustable sensitivity parameters
                normalized = (amplitudes - self.min_db) / (self.max_db - self.min_db)
                normalized = np.clip(normalized, 0, 1)

                # Apply power curve for more dramatic effect
                normalized = np.power(normalized, 0.6)  # Makes bars jump more (lower = more dramatic)

                # Apply smoothing for more fluid animation
                target_heights = normalized * (self.height * 0.98)  # Use almost full screen height
                self.bar_heights = (self.bar_smoothing * self.bar_heights +
                                   (1 - self.bar_smoothing) * target_heights)

                # Update peaks
                for i in range(self.num_bars):
                    if self.bar_heights[i] > self.peak_heights[i]:
                        # New peak reached
                        self.peak_heights[i] = self.bar_heights[i]
                        self.peak_hold_time[i] = 30  # Hold for 30 frames (~0.5 seconds at 60fps)
                    else:
                        # Decrease hold time
                        if self.peak_hold_time[i] > 0:
                            self.peak_hold_time[i] -= 1
                        else:
                            # Peak falls down
                            self.peak_heights[i] -= self.peak_fall_speed
                            if self.peak_heights[i] < self.bar_heights[i]:
                                self.peak_heights[i] = self.bar_heights[i]

            except queue.Empty:
                break

    def _update_starfield(self):
        """Update star positions with parallax scrolling"""
        # Check if we should draw trails (warp speed at 80%+)
        warp_speed = self.starfield_speed >= 2.4  # 80% of max (3.0)

        for star in self.stars:
            # Move star to the left with speed multiplier
            actual_speed = star.speed * self.starfield_speed

            # Store trail positions for warp effect
            if warp_speed:
                # Add current position to trail
                star.trail_positions.append((star.x, star.y))
                # Keep trail length based on speed (faster = longer trails)
                max_trail_length = int(10 + (self.starfield_speed - 2.4) * 20)  # 10-22 positions
                if len(star.trail_positions) > max_trail_length:
                    star.trail_positions.pop(0)
            else:
                # Clear trails when not at warp speed
                star.trail_positions = []

            star.x -= actual_speed

            # Wrap around when star goes off left edge
            if star.x < -10:
                star.x = self.width + 10
                star.y = random.uniform(0, self.height)
                star.trail_positions = []  # Clear trail on wrap

            # Update twinkling (slower at high speeds)
            twinkle_multiplier = 1.0 if self.starfield_speed < 1.5 else 0.3
            star.brightness += star.twinkle_speed * star.twinkle_direction * twinkle_multiplier
            if star.brightness >= 1.0:
                star.brightness = 1.0
                star.twinkle_direction = -1
            elif star.brightness <= 0.3:
                star.brightness = 0.3
                star.twinkle_direction = 1

    def _update_menu_visibility(self):
        """Update menu visibility based on mouse position and time"""
        _, mouse_y = pygame.mouse.get_pos()
        current_time = pygame.time.get_ticks()

        # Show menu if mouse is near the top of the screen
        if mouse_y <= self.menu_show_threshold:
            if not self.menu_visible:
                self.menu_visible = True
            self.last_mouse_move_time = current_time
        else:
            # Hide menu if it's been inactive for too long and no dropdown is open
            time_since_move = current_time - self.last_mouse_move_time
            if time_since_move > self.menu_hide_delay and self.open_dropdown is None:
                self.menu_visible = False

    def _draw_star(self, star, glass_y):
        """Draw a star with lens flare effect and warp streaks"""
        # Only draw stars above the glass horizon (or full height if reflection disabled)
        if star.y > glass_y:
            return

        # Apply brightness/twinkling
        color = tuple(int(c * star.brightness) for c in star.color)

        # Draw warp streak/trail if we have trail positions
        if len(star.trail_positions) > 1:
            trail_count = len(star.trail_positions)
            for i in range(trail_count - 1):
                # Fade from back of trail (oldest) to front (newest)
                fade = (i + 1) / trail_count
                alpha = int(255 * star.brightness * fade * 0.6)  # Max 60% opacity for trails
                trail_color = (*color, alpha)

                # Draw trail segment
                pos1 = star.trail_positions[i]
                pos2 = star.trail_positions[i + 1]

                # Only draw if both positions are above glass
                if pos1[1] <= glass_y and pos2[1] <= glass_y:
                    # Create surface for alpha blending
                    # Calculate bounding box for the line
                    min_x = min(pos1[0], pos2[0]) - 2
                    max_x = max(pos1[0], pos2[0]) + 2
                    min_y = min(pos1[1], pos2[1]) - 2
                    max_y = max(pos1[1], pos2[1]) + 2
                    width = max(1, int(max_x - min_x))
                    height = max(1, int(max_y - min_y))

                    surf = pygame.Surface((width, height), pygame.SRCALPHA)
                    # Adjust coordinates relative to surface
                    rel_pos1 = (pos1[0] - min_x, pos1[1] - min_y)
                    rel_pos2 = (pos2[0] - min_x, pos2[1] - min_y)
                    pygame.draw.line(surf, trail_color, rel_pos1, rel_pos2, max(1, int(star.size * 0.5)))
                    self.screen.blit(surf, (min_x, min_y))

        # Draw lens flare cross effect for larger stars
        if star.size >= 1.5:
            # Horizontal beam
            beam_length = int(star.size * 3)
            pygame.draw.line(self.screen, color,
                           (star.x - beam_length, star.y),
                           (star.x + beam_length, star.y), 1)
            # Vertical beam
            pygame.draw.line(self.screen, color,
                           (star.x, star.y - beam_length),
                           (star.x, star.y + beam_length), 1)

        # Draw star center glow (multiple circles for glow effect)
        glow_radius = max(1, int(star.size))
        for i in range(glow_radius, 0, -1):
            alpha = int(255 * star.brightness * (i / glow_radius))
            glow_color = (*color, alpha)
            surf = pygame.Surface((i*2+2, i*2+2), pygame.SRCALPHA)
            pygame.draw.circle(surf, glow_color, (i+1, i+1), i)
            self.screen.blit(surf, (star.x - i - 1, star.y - i - 1))

        # Bright center point
        pygame.draw.circle(self.screen, (255, 255, 255), (int(star.x), int(star.y)), 1)

    def _draw_star_reflection(self, star, glass_y):
        """Draw reflected star below the glass horizon"""
        if star.y > glass_y:
            return  # Star is below horizon, don't reflect

        # Calculate reflection position
        distance_from_glass = glass_y - star.y
        reflection_y = glass_y + distance_from_glass

        # Don't draw if reflection is off screen
        if reflection_y > self.height:
            return

        # Fade reflection based on distance from glass
        max_reflection_height = self.height - glass_y
        fade_ratio = 1.0 - (distance_from_glass / max_reflection_height)
        fade_ratio = max(0, min(1, fade_ratio)) ** 1.5

        # Apply reflection intensity and fade
        opacity = self.reflection_intensity * fade_ratio * star.brightness

        if opacity < 0.01:
            return

        # Dimmed and desaturated color for reflection
        color = tuple(int(c * 0.7 * opacity) for c in star.color)

        # Draw dimmer lens flare for reflection
        if star.size >= 1.5:
            beam_length = int(star.size * 2)
            alpha_color = (*color, int(255 * opacity))
            surf_h = pygame.Surface((beam_length*2+2, 3), pygame.SRCALPHA)
            pygame.draw.line(surf_h, alpha_color,
                           (0, 1), (beam_length*2, 1), 1)
            self.screen.blit(surf_h, (star.x - beam_length, reflection_y - 1))

            surf_v = pygame.Surface((3, beam_length*2+2), pygame.SRCALPHA)
            pygame.draw.line(surf_v, alpha_color,
                           (1, 0), (1, beam_length*2), 1)
            self.screen.blit(surf_v, (star.x - 1, reflection_y - beam_length))

        # Draw reflection glow
        glow_radius = max(1, int(star.size * 0.8))
        for i in range(glow_radius, 0, -1):
            alpha = int(255 * opacity * (i / glow_radius))
            glow_color = (*color, alpha)
            surf = pygame.Surface((i*2+2, i*2+2), pygame.SRCALPHA)
            pygame.draw.circle(surf, glow_color, (i+1, i+1), i)
            self.screen.blit(surf, (star.x - i - 1, reflection_y - i - 1))

    def draw(self):
        """Draw the visualization"""
        # Clear screen with black background
        self.screen.fill((0, 0, 0))

        # Calculate layout - if reflection is enabled, split the visualizer area
        if self.reflection_enabled:
            # Glass surface at 60% down the visualizer height
            glass_y = int(self.height * 0.6)
            visualizer_height = glass_y  # Bars only go up to glass surface
        else:
            glass_y = self.height
            visualizer_height = self.height

        # Draw starfield behind everything (sorted by layer for proper depth)
        if self.starfield_enabled:
            # Sort stars by layer (back to front)
            sorted_stars = sorted(self.stars, key=lambda s: s.layer)
            for star in sorted_stars:
                self._draw_star(star, glass_y)

        # Calculate bar dimensions to fill visualizer width (not full width)
        bar_spacing = 2
        total_bars_space = self.visualizer_width - (bar_spacing * (self.num_bars - 1))
        bar_width = total_bars_space / self.num_bars

        # Scale bar heights to fit in the available visualizer height
        height_scale = visualizer_height / self.height if self.reflection_enabled else 1.0

        # Draw each frequency bar
        for i in range(self.num_bars):
            x = i * (bar_width + bar_spacing)
            bar_height = self.bar_heights[i] * height_scale
            y = glass_y - bar_height

            # Draw the main bar (grows upward from glass surface)
            self._draw_bar(x, y, bar_width, bar_height, i, base_y=glass_y)

            # Draw the peak indicator
            peak_height = self.peak_heights[i] * height_scale
            peak_y = glass_y - peak_height
            self._draw_peak(x, peak_y, bar_width, i)

        # Draw glass surface line (if reflection enabled)
        if self.reflection_enabled:
            # Subtle horizontal line at glass surface
            pygame.draw.line(self.screen, (80, 80, 80),
                            (0, glass_y),
                            (self.visualizer_width, glass_y), 1)

            # Create a temporary surface for reflections
            reflection_height = self.height - glass_y
            reflection_surface = pygame.Surface((self.visualizer_width, reflection_height), pygame.SRCALPHA)
            reflection_surface.fill((0, 0, 0, 0))  # Transparent background

            # Temporarily redirect drawing to reflection surface
            original_screen = self.screen
            self.screen = reflection_surface

            # Draw reflections to the temporary surface (adjust Y coordinates)
            for i in range(self.num_bars):
                x = i * (bar_width + bar_spacing)
                bar_height = self.bar_heights[i] * height_scale

                # Mirror position starts at 0 on reflection surface (which is glass_y on main screen)
                mirror_y = 0

                # Draw reflected bar
                self._draw_bar_reflection(x, mirror_y, bar_width, bar_height, i, glass_y=0)

                # Draw reflected peak
                peak_height = self.peak_heights[i] * height_scale
                peak_mirror_y = peak_height
                self._draw_peak_reflection(x, peak_mirror_y, bar_width, i, glass_y=0)

            # Restore original screen
            self.screen = original_screen

            # Apply blur to the reflection surface
            if self.reflection_blur > 0:
                reflection_surface = self._apply_blur(reflection_surface, self.reflection_blur)

            # Blit the blurred reflection surface to the main screen
            self.screen.blit(reflection_surface, (0, glass_y))

            # Draw star reflections (after bar reflections)
            if self.starfield_enabled:
                for star in sorted_stars:
                    self._draw_star_reflection(star, glass_y)

        # Draw menu bar (if visible)
        if self.menu_visible:
            self._draw_menu_bar()

        pygame.display.flip()

    def _draw_menu_bar(self):
        """Draw the top menu bar with dropdown menus"""
        # Draw menu bar background
        menu_bar_rect = pygame.Rect(0, 0, self.width, self.menu_height)
        pygame.draw.rect(self.screen, (30, 30, 30), menu_bar_rect)
        pygame.draw.line(self.screen, (60, 60, 60),
                        (0, self.menu_height),
                        (self.width, self.menu_height), 1)

        # Draw menu items
        for menu_item in self.menu_items:
            # Highlight if this menu is open
            is_open = (self.open_dropdown == menu_item['name'])

            # Menu item background
            if is_open:
                color = (50, 50, 50)
            else:
                color = (30, 30, 30)

            menu_rect = pygame.Rect(menu_item['x'], menu_item['y'],
                                   menu_item['width'], menu_item['height'])
            pygame.draw.rect(self.screen, color, menu_rect)
            pygame.draw.rect(self.screen, (80, 80, 80), menu_rect, 1)

            # Menu item text
            text_surface = self.font.render(menu_item['name'], True, (220, 220, 220))
            text_rect = text_surface.get_rect(center=(
                menu_item['x'] + menu_item['width'] // 2,
                menu_item['y'] + menu_item['height'] // 2
            ))
            self.screen.blit(text_surface, text_rect)

            # Draw dropdown arrow
            arrow_text = self.font_small.render('▼', True, (150, 150, 150))
            arrow_x = menu_item['x'] + menu_item['width'] - 15
            arrow_y = menu_item['y'] + menu_item['height'] // 2 - 5
            self.screen.blit(arrow_text, (arrow_x, arrow_y))

            # Draw dropdown panel if this menu is open
            if is_open:
                self._draw_dropdown_panel(menu_item)

    def _draw_dropdown_panel(self, menu_item):
        """Draw dropdown panel with controls for a menu item"""
        # Calculate panel dimensions
        panel_padding = 10
        panel_x = menu_item['x']
        panel_y = menu_item['y'] + menu_item['height']
        panel_width = 220

        # Calculate panel height based on visible controls
        visible_controls = []
        for control in menu_item['controls']:
            # Check visibility conditions
            if 'visible_when' in control:
                if control['visible_when'] == 'reflection_enabled' and not self.reflection_enabled:
                    continue
                if control['visible_when'] == 'starfield_enabled' and not self.starfield_enabled:
                    continue
            visible_controls.append(control)

        # Calculate height
        control_spacing = 50
        panel_height = panel_padding * 2 + len(visible_controls) * control_spacing

        # Draw panel background
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, (40, 40, 40), panel_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), panel_rect, 2)

        # Draw controls inside the panel
        y_offset = panel_y + panel_padding
        for control in visible_controls:
            # Set control position
            control['x'] = panel_x + 15
            control['y'] = y_offset

            # Draw the control
            if control['type'] == 'slider':
                self._draw_slider(control)
            elif control['type'] == 'button':
                self._draw_button(control)

            y_offset += control_spacing

    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode"""
        # Prevent resize events from interfering
        self.ignore_resize = True

        self.is_fullscreen = not self.is_fullscreen

        if self.is_fullscreen:
            print("Switching to fullscreen mode...")
            print(f"Current dimensions before fullscreen: {self.width}x{self.height}")

            # Try to get display info
            try:
                # Get all available fullscreen modes
                modes = pygame.display.list_modes()
                print(f"Available display modes: {modes[:5]}...")  # Show first 5

                # Use the largest available mode (native resolution)
                if modes and modes[0] != -1:
                    screen_width, screen_height = modes[0]
                    print(f"Using highest resolution: {screen_width}x{screen_height}")
                else:
                    # Fallback to display info
                    display_info = pygame.display.Info()
                    screen_width = display_info.current_w
                    screen_height = display_info.current_h
                    print(f"Fallback to display info: {screen_width}x{screen_height}")

                # Use pygame's built-in fullscreen mode with explicit resolution
                self.screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)

                # Update dimensions to fullscreen
                self.width = screen_width
                self.height = screen_height
                print(f"Fullscreen mode set: {self.width}x{self.height}")
            except Exception as e:
                print(f"Error setting fullscreen: {e}")
                self.is_fullscreen = False
        else:
            print("Switching to windowed mode...")
            print(f"Restoring to original dimensions: {self.original_width}x{self.original_height}")

            # Always restore to the original launch dimensions
            self.screen = pygame.display.set_mode((self.original_width, self.original_height), pygame.RESIZABLE)

            # Set our dimensions back to original
            self.width = self.original_width
            self.height = self.original_height
            print(f"Windowed mode set: {self.width}x{self.height}")

        # Update visualizer width (now full width)
        self.visualizer_width = self.width

        # Re-enable resize events after a short delay (let pending events clear)
        self.ignore_resize_until = pygame.time.get_ticks() + 500  # Ignore for 500ms

    def _handle_mouse_click(self, mouse_pos):
        """Handle mouse clicks on menu items and controls"""
        mouse_x, mouse_y = mouse_pos

        # Check if click is on menu bar
        if self.menu_visible and mouse_y <= self.menu_height:
            clicked_menu = False
            for menu_item in self.menu_items:
                if (menu_item['x'] <= mouse_x <= menu_item['x'] + menu_item['width'] and
                    menu_item['y'] <= mouse_y <= menu_item['y'] + menu_item['height']):
                    # Toggle dropdown for this menu
                    if self.open_dropdown == menu_item['name']:
                        self.open_dropdown = None
                    else:
                        self.open_dropdown = menu_item['name']
                    clicked_menu = True
                    self.last_mouse_move_time = pygame.time.get_ticks()  # Keep menu visible
                    break

            if not clicked_menu:
                # Clicked on menu bar but not on a menu item - close dropdowns
                self.open_dropdown = None
            return

        # Check if click is on an open dropdown panel
        if self.open_dropdown is not None:
            for menu_item in self.menu_items:
                if menu_item['name'] == self.open_dropdown:
                    # Calculate dropdown panel bounds
                    panel_x = menu_item['x']
                    panel_y = menu_item['y'] + menu_item['height']
                    panel_width = 220

                    # Calculate visible controls
                    visible_controls = []
                    for control in menu_item['controls']:
                        if 'visible_when' in control:
                            if control['visible_when'] == 'reflection_enabled' and not self.reflection_enabled:
                                continue
                            if control['visible_when'] == 'starfield_enabled' and not self.starfield_enabled:
                                continue
                        visible_controls.append(control)

                    control_spacing = 50
                    panel_height = 20 + len(visible_controls) * control_spacing

                    # Check if click is inside dropdown panel
                    if (panel_x <= mouse_x <= panel_x + panel_width and
                        panel_y <= mouse_y <= panel_y + panel_height):
                        # Check which control was clicked
                        y_offset = panel_y + 10
                        for control in visible_controls:
                            control['x'] = panel_x + 15
                            control['y'] = y_offset

                            if control['type'] == 'slider':
                                if (control['x'] <= mouse_x <= control['x'] + control['width'] and
                                    control['y'] <= mouse_y <= control['y'] + control['height']):
                                    self.dragging_control = control
                                    self._handle_slider_drag(control, mouse_x)
                                    self.last_mouse_move_time = pygame.time.get_ticks()
                                    return
                            elif control['type'] == 'button':
                                if (control['x'] <= mouse_x <= control['x'] + control['width'] and
                                    control['y'] <= mouse_y <= control['y'] + control['height']):
                                    self._handle_button_click(control)
                                    self.last_mouse_move_time = pygame.time.get_ticks()
                                    return

                            y_offset += control_spacing
                        self.last_mouse_move_time = pygame.time.get_ticks()
                        return
                    else:
                        # Clicked outside dropdown - close it
                        self.open_dropdown = None
                        return

        # Click somewhere else - close any open dropdowns
        self.open_dropdown = None

    def handle_resize(self, new_width, new_height):
        """Handle window resize"""
        # Ignore resize events during and shortly after fullscreen toggle
        if hasattr(self, 'ignore_resize_until') and pygame.time.get_ticks() < self.ignore_resize_until:
            print(f"Ignoring resize event: {new_width}x{new_height}")
            return

        print(f"Handling resize to: {new_width}x{new_height}")
        self.width = new_width
        self.height = new_height
        self.visualizer_width = self.width  # Full width now
        self.screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)

    def run(self):
        """Main loop"""
        clock = pygame.time.Clock()

        print("Audio Visualizer started!")
        print("Capturing system audio... Play some music to see the visualization!")
        print("Press F11 to toggle fullscreen mode.")
        print("Press ESC or close the window to exit.")

        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_F11:
                        # F11 toggles fullscreen
                        self.toggle_fullscreen()
                elif event.type == pygame.VIDEORESIZE:
                    self.handle_resize(event.w, event.h)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self._handle_mouse_click(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left click release
                        self.dragging_control = None
                elif event.type == pygame.MOUSEMOTION:
                    # Update last mouse move time for menu visibility
                    if event.pos[1] <= self.menu_show_threshold:
                        self.last_mouse_move_time = pygame.time.get_ticks()
                    # Handle slider dragging
                    if self.dragging_control is not None:
                        self._handle_slider_drag(self.dragging_control, event.pos[0])

            # Update and draw
            self.update()
            self.draw()

            # Cap at 60 FPS
            clock.tick(60)

        pygame.quit()
        sys.exit()

def main():
    visualizer = AudioVisualizer(width=1280, height=720, num_bars=38)
    visualizer.run()

if __name__ == "__main__":
    main()
