#!/usr/bin/env python3
"""
Test script for PIL and Tkinter integration.
This script helps diagnose issues with PIL/Pillow and Tkinter on macOS.
"""

import os
import sys
import platform
import tkinter as tk
from tkinter import ttk, messagebox

# Print system information
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"System: {platform.system()}")
print(f"Python executable: {sys.executable}")

# Try to import PIL/Pillow
try:
    import PIL
    from PIL import Image, ImageTk
    print(f"PIL version: {PIL.__version__}")
    print("PIL import successful")
    PIL_AVAILABLE = True
except (ImportError, TypeError) as e:
    print(f"PIL import error: {e}")
    PIL_AVAILABLE = False

# Create a simple Tkinter window
root = tk.Tk()
root.title("PIL/Tkinter Test")
root.geometry("400x300")

# Add a label with system info
info_text = f"Python: {sys.version.split()[0]}\nPlatform: {platform.system()}\nPIL available: {PIL_AVAILABLE}"
info_label = ttk.Label(root, text=info_text, padding=10)
info_label.pack(pady=10)

# Try to create and display a simple image if PIL is available
if PIL_AVAILABLE:
    try:
        # Create a simple colored image
        img = Image.new('RGB', (200, 100), color='red')
        
        # Try to convert it to a Tkinter-compatible format
        try:
            photo = ImageTk.PhotoImage(img)
            print("Successfully created PhotoImage")
            
            # Display the image
            img_label = ttk.Label(root, image=photo)
            img_label.image = photo  # Keep a reference
            img_label.pack(pady=10)
            
            status_label = ttk.Label(root, text="PIL/Tkinter integration working correctly", foreground="green")
            status_label.pack(pady=10)
        except Exception as e:
            print(f"Error creating PhotoImage: {e}")
            error_label = ttk.Label(root, text=f"PIL/Tkinter integration error:\n{str(e)}", foreground="red")
            error_label.pack(pady=10)
    except Exception as e:
        print(f"Error creating test image: {e}")
        error_label = ttk.Label(root, text=f"Error creating test image:\n{str(e)}", foreground="red")
        error_label.pack(pady=10)
else:
    error_label = ttk.Label(root, text="PIL/Pillow not available", foreground="red")
    error_label.pack(pady=10)

# Add a button to close the window
close_button = ttk.Button(root, text="Close", command=root.destroy)
close_button.pack(pady=20)

# Start the Tkinter event loop
root.mainloop() 