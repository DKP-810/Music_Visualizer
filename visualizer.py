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

        # Create resizable windowed mode (explicitly NOT fullscreen)
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("Audio Visualizer")

        # Audio settings
        self.sample_rate = 48000
        self.chunk_size = 2048
        self.audio_queue = queue.Queue()

        # Visualization data
        self.bar_heights = np.zeros(num_bars)
        self.peak_heights = np.zeros(num_bars)
        self.peak_hold_time = np.zeros(num_bars)
        self.peak_fall_speed = 0.5  # Speed at which peaks fall
        self.bar_smoothing = 0.7  # Smoothing factor for bar movements

        # Colors - Rainbow gradient
        self.colors = self._generate_rainbow_gradient()

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
                # Adjust these values to change sensitivity
                min_db = 20  # Minimum dB to display (lower = more sensitive)
                max_db = 75  # Maximum dB for full height (lower = more sensitive)

                normalized = (amplitudes - min_db) / (max_db - min_db)
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

        # Calculate bar dimensions to fill entire width
        bar_spacing = 2
        total_bars_space = self.width - (bar_spacing * (self.num_bars - 1))
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

        pygame.display.flip()

    def handle_resize(self, new_width, new_height):
        """Handle window resize"""
        self.width = new_width
        self.height = new_height
        self.screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)

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
