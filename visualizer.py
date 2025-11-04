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

class AudioVisualizer:
    def __init__(self, width=1280, height=720, num_bars=38):
        pygame.init()

        # Window settings
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

        # Control panel settings
        self.panel_width = 200
        self.visualizer_width = self.width - self.panel_width

        # Create resizable windowed mode (explicitly NOT fullscreen)
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("Audio Visualizer")

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

        # Visualization data
        self.bar_heights = np.zeros(num_bars)
        self.peak_heights = np.zeros(num_bars)
        self.peak_hold_time = np.zeros(num_bars)
        self.peak_fall_speed = 0.5  # Speed at which peaks fall
        self.bar_smoothing = 0.7  # Smoothing factor for bar movements

        # Colors - Rainbow gradient
        self.colors = self._generate_rainbow_gradient()

        # UI Controls
        self.controls = self._create_controls()
        self.dragging_control = None

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

    def _create_controls(self):
        """Create UI control elements"""
        controls = []
        panel_x = self.visualizer_width + 10
        y_offset = 20

        # Bar Count Slider
        controls.append({
            'type': 'slider',
            'name': 'Bars',
            'x': panel_x,
            'y': y_offset,
            'width': 180,
            'height': 20,
            'min': 10,
            'max': 100,
            'value': self.num_bars,
            'param': 'num_bars'
        })
        y_offset += 60

        # Sensitivity Slider (inverted - higher slider value = more sensitive)
        controls.append({
            'type': 'slider',
            'name': 'Sensitivity',
            'x': panel_x,
            'y': y_offset,
            'width': 180,
            'height': 20,
            'min': 10,
            'max': 50,
            'value': 60 - self.min_db,  # Inverted display value
            'param': 'min_db'
        })
        y_offset += 60

        # Color Scheme Toggle Button
        controls.append({
            'type': 'button',
            'name': 'Color Scheme',
            'x': panel_x,
            'y': y_offset,
            'width': 180,
            'height': 40,
            'param': 'color_scheme',
            'states': ['rainbow', 'green'],
            'state_index': 0
        })

        return controls

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

        # Button background
        current_state = control['states'][control['state_index']]
        if current_state == 'rainbow':
            color = (100, 50, 150)
        else:
            color = (0, 100, 50)

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

    def _handle_button_click(self, control):
        """Handle clicking a button"""
        if control['param'] == 'color_scheme':
            # Toggle between states
            control['state_index'] = (control['state_index'] + 1) % len(control['states'])
            self.color_scheme = control['states'][control['state_index']]
            self._update_colors()

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

    def _draw_bar(self, x, y, width, height, bar_index):
        """Draw a segmented frequency bar with retro glow effect"""
        if height < 1:
            return

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
            seg_y = self.height - ((seg + 1) * total_segment_height)

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

    def update(self):
        """Update visualization with latest audio data"""
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

    def draw(self):
        """Draw the visualization"""
        # Clear screen with black background
        self.screen.fill((0, 0, 0))

        # Draw control panel background
        panel_rect = pygame.Rect(self.visualizer_width, 0, self.panel_width, self.height)
        pygame.draw.rect(self.screen, (20, 20, 20), panel_rect)
        pygame.draw.line(self.screen, (60, 60, 60),
                        (self.visualizer_width, 0),
                        (self.visualizer_width, self.height), 2)

        # Calculate bar dimensions to fill visualizer width (not full width)
        bar_spacing = 2
        total_bars_space = self.visualizer_width - (bar_spacing * (self.num_bars - 1))
        bar_width = total_bars_space / self.num_bars

        # Draw each frequency bar
        for i in range(self.num_bars):
            x = i * (bar_width + bar_spacing)
            bar_height = self.bar_heights[i]
            y = self.height - bar_height

            # Draw the main bar
            self._draw_bar(x, y, bar_width, bar_height, i)

            # Draw the peak indicator
            peak_y = self.height - self.peak_heights[i]
            self._draw_peak(x, peak_y, bar_width, i)

        # Draw controls
        for control in self.controls:
            if control['type'] == 'slider':
                self._draw_slider(control)
            elif control['type'] == 'button':
                self._draw_button(control)

        pygame.display.flip()

    def handle_resize(self, new_width, new_height):
        """Handle window resize"""
        self.width = new_width
        self.height = new_height
        self.visualizer_width = self.width - self.panel_width
        self.screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)

        # Update control positions
        panel_x = self.visualizer_width + 10
        y_offset = 20
        for control in self.controls:
            control['x'] = panel_x
            control['y'] = y_offset
            y_offset += 60 if control['type'] == 'slider' else 70

    def run(self):
        """Main loop"""
        clock = pygame.time.Clock()

        print("Audio Visualizer started!")
        print("Capturing system audio... Play some music to see the visualization!")
        print("Press ESC or close the window to exit.")

        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.handle_resize(event.w, event.h)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        mouse_x, mouse_y = event.pos
                        # Check if click is on any control
                        for control in self.controls:
                            if control['type'] == 'slider':
                                if (control['x'] <= mouse_x <= control['x'] + control['width'] and
                                    control['y'] <= mouse_y <= control['y'] + control['height']):
                                    self.dragging_control = control
                                    self._handle_slider_drag(control, mouse_x)
                            elif control['type'] == 'button':
                                if (control['x'] <= mouse_x <= control['x'] + control['width'] and
                                    control['y'] <= mouse_y <= control['y'] + control['height']):
                                    self._handle_button_click(control)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left click release
                        self.dragging_control = None
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging_control is not None:
                        mouse_x, mouse_y = event.pos
                        self._handle_slider_drag(self.dragging_control, mouse_x)

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
