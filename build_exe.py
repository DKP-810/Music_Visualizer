"""
Build script to create a standalone executable using PyInstaller
"""

import os
import sys
import subprocess

def build_executable():
    """Build the executable using PyInstaller"""

    print("Building Audio Visualizer executable...")
    print("-" * 50)

    # PyInstaller command
    command = [
        "pyinstaller",
        "--onefile",  # Single executable file
        "--windowed",  # No console window (remove this if you want to see debug output)
        "--name=AudioVisualizer",
        "--icon=NONE",  # You can add an .ico file here later
        "visualizer.py"
    ]

    # For debugging, use this instead (shows console):
    # command = [
    #     "pyinstaller",
    #     "--onefile",
    #     "--name=AudioVisualizer",
    #     "visualizer.py"
    # ]

    try:
        subprocess.run(command, check=True)
        print("\n" + "=" * 50)
        print("Build successful!")
        print("=" * 50)
        print(f"\nExecutable location: dist\\AudioVisualizer.exe")
        print("\nYou can now run the executable without Python installed!")
        print("Note: First run may take a few seconds to start.")

    except subprocess.CalledProcessError as e:
        print(f"\nBuild failed with error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: PyInstaller not found!")
        print("Please install it with: pip install pyinstaller")
        sys.exit(1)

if __name__ == "__main__":
    build_executable()
