#!/usr/bin/env python3
"""
GUI Application for Facial Recognition.
This application provides a graphical user interface for the facial recognition system.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import threading
import time
import numpy as np
import platform
import traceback
import sys
from utils import load_known_faces, detect_and_display_faces, load_training_data, save_known_faces

# Try to import face_recognition
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: face_recognition import error: {e}")
    FACE_RECOGNITION_AVAILABLE = False

# Try to import PIL/Pillow modules with error handling
try:
    import PIL.Image, PIL.ImageTk
    PIL_AVAILABLE = True
except (ImportError, TypeError) as e:
    print(f"Warning: PIL/Pillow import error: {e}")
    PIL_AVAILABLE = False

# Set up Info.plist for macOS
if platform.system() == 'Darwin':
    # Check if Info.plist exists in the current directory
    if os.path.exists('Info.plist'):
        # Set the environment variable to use the Info.plist file
        os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
        
        # Suppress the AVCaptureDevice warning
        os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'
        
        # Print a message about using Info.plist
        print("Using Info.plist for macOS camera configuration")

class FacialRecognitionApp:
    def __init__(self, window, window_title):
        # Initialize the window
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1000x600")
        self.window.resizable(True, True)
        
        # Set window icon if available
        # self.window.iconbitmap('icon.ico')  # Uncomment and add an icon file if desired
        
        # Initialize variables
        self.camera_index = 0
        self.recognition_threshold = 0.6
        self.scale_factor = 0.5  # Default scale factor for face detection
        self.performance_mode = True  # Default to performance mode
        self.model_path = "known_faces.pkl"
        self.training_dir = "training_data"
        self.is_running = False
        self.thread = None
        self.stopEvent = None
        self.video_capture = None
        self.photo = None
        self.use_pil = PIL_AVAILABLE
        
        # Check if running on macOS
        self.is_macos = platform.system() == 'Darwin'
        
        # Load known faces
        self.known_face_encodings, self.known_face_names = load_known_faces(self.model_path)
        
        # Create the main frame
        self.main_frame = ttk.Frame(window)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create the left panel (video feed)
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create the right panel (controls)
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        # Create the video canvas
        self.canvas = tk.Canvas(self.left_panel, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create the control panel
        self.create_control_panel()
        
        # Create the status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(window, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Set the window close handler
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Update the status
        self.update_status()
        
        # Display PIL availability status
        if not self.use_pil:
            self.status_var.set("Warning: PIL/Pillow not fully available. Using fallback display method.")
            messagebox.showwarning("PIL/Pillow Issue", 
                                  "There seems to be an issue with PIL/Pillow integration with Tkinter. "
                                  "The application will use a fallback method to display images, "
                                  "which might be slower or lower quality.")
    
    def create_control_panel(self):
        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create the camera tab
        self.camera_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.camera_tab, text="Camera")
        
        # Create the training tab
        self.training_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.training_tab, text="Training")
        
        # Create the settings tab
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text="Settings")
        
        # Populate the camera tab
        self.populate_camera_tab()
        
        # Populate the training tab
        self.populate_training_tab()
        
        # Populate the settings tab
        self.populate_settings_tab()
    
    def populate_camera_tab(self):
        # Camera controls frame
        camera_frame = ttk.LabelFrame(self.camera_tab, text="Camera Controls")
        camera_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Single Capture & Analyze button
        self.capture_btn = ttk.Button(camera_frame, text="Capture & Analyze", command=self.single_capture_and_analyze)
        self.capture_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Camera selection
        camera_select_frame = ttk.Frame(camera_frame)
        camera_select_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(camera_select_frame, text="Camera:").pack(side=tk.LEFT)
        self.camera_var = tk.IntVar(value=self.camera_index)
        camera_spinbox = ttk.Spinbox(camera_select_frame, from_=0, to=10, textvariable=self.camera_var, width=5)
        camera_spinbox.pack(side=tk.LEFT, padx=5)
        
        # macOS camera permission note
        if self.is_macos:
            mac_note = ttk.Label(camera_frame, text="Note: On macOS, you may need to grant camera permissions.", 
                                 foreground="red", wraplength=200)
            mac_note.pack(fill=tk.X, padx=5, pady=5)
        
        # Recognition info frame
        recognition_frame = ttk.LabelFrame(self.camera_tab, text="Recognition Info")
        recognition_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Recognition status
        self.recognition_status = tk.Text(recognition_frame, height=10, width=30, state=tk.DISABLED)
        self.recognition_status.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize the canvas with a placeholder
        self.canvas.delete("all")
        
        # Create a nicer placeholder with instructions
        self.canvas.create_rectangle(0, 0, 1000, 1000, fill="black")
        
        # Add a camera icon or placeholder text
        self.canvas.create_text(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2 - 40,
            text="📷",
            fill="white",
            font=("Arial", 48)
        )
        
        self.canvas.create_text(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2 + 40,
            text="Click 'Capture & Analyze' to take a photo",
            fill="white",
            font=("Arial", 14)
        )
        
        self.canvas.create_text(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2 + 70,
            text="High-resolution image will be captured and analyzed",
            fill="white",
            font=("Arial", 12)
        )
    
    def populate_training_tab(self):
        # Training controls frame
        training_frame = ttk.LabelFrame(self.training_tab, text="Training Controls")
        training_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Training directory
        dir_frame = ttk.Frame(training_frame)
        dir_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(dir_frame, text="Training Directory:").pack(side=tk.LEFT)
        self.training_dir_var = tk.StringVar(value=self.training_dir)
        training_dir_entry = ttk.Entry(dir_frame, textvariable=self.training_dir_var, width=20)
        training_dir_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        browse_btn = ttk.Button(dir_frame, text="Browse...", command=self.browse_training_dir)
        browse_btn.pack(side=tk.LEFT, padx=5)
        
        # Train button
        train_btn = ttk.Button(training_frame, text="Train Model", command=self.train_model)
        train_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Add person frame
        add_person_frame = ttk.LabelFrame(self.training_tab, text="Add Person")
        add_person_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Person name
        name_frame = ttk.Frame(add_person_frame)
        name_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(name_frame, text="Name:").pack(side=tk.LEFT)
        self.person_name_var = tk.StringVar()
        person_name_entry = ttk.Entry(name_frame, textvariable=self.person_name_var, width=20)
        person_name_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Capture button
        capture_btn = ttk.Button(add_person_frame, text="Capture Image", command=self.capture_image)
        capture_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Training status
        self.training_status = tk.Text(self.training_tab, height=10, width=30, state=tk.DISABLED)
        self.training_status.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def populate_settings_tab(self):
        # Settings frame
        settings_frame = ttk.LabelFrame(self.settings_tab, text="Recognition Settings")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Threshold
        threshold_frame = ttk.Frame(settings_frame)
        threshold_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(threshold_frame, text="Recognition Threshold:").pack(side=tk.LEFT)
        self.threshold_var = tk.DoubleVar(value=self.recognition_threshold)
        threshold_scale = ttk.Scale(threshold_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, 
                                   variable=self.threshold_var, length=200)
        threshold_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        threshold_label = ttk.Label(threshold_frame, textvariable=self.threshold_var, width=5)
        threshold_label.pack(side=tk.LEFT, padx=5)
        
        # Scale factor for face detection
        scale_frame = ttk.Frame(settings_frame)
        scale_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(scale_frame, text="Detection Scale:").pack(side=tk.LEFT)
        self.scale_factor_var = tk.DoubleVar(value=self.scale_factor)
        scale_factor_scale = ttk.Scale(scale_frame, from_=0.2, to=1.0, orient=tk.HORIZONTAL, 
                                      variable=self.scale_factor_var, length=200)
        scale_factor_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        scale_factor_label = ttk.Label(scale_frame, textvariable=self.scale_factor_var, width=5)
        scale_factor_label.pack(side=tk.LEFT, padx=5)
        
        # Display quality frame
        quality_frame = ttk.LabelFrame(settings_frame, text="Display Quality")
        quality_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Performance mode checkbox
        perf_frame = ttk.Frame(quality_frame)
        perf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.performance_mode_var = tk.BooleanVar(value=self.performance_mode)
        performance_mode_check = ttk.Checkbutton(perf_frame, text="Performance Mode", 
                                                variable=self.performance_mode_var)
        performance_mode_check.pack(side=tk.LEFT, padx=5)
        
        # Add a description of performance mode
        performance_description = ttk.Label(quality_frame, 
                                          text="Performance Mode optimizes for speed over quality.\n"
                                               "• Enabled: Faster processing, lower resolution display\n"
                                               "• Disabled: Higher quality display, may be slower",
                                          wraplength=300, justify=tk.LEFT)
        performance_description.pack(fill=tk.X, padx=5, pady=5)
        
        # Model path
        model_frame = ttk.Frame(settings_frame)
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Model File:").pack(side=tk.LEFT)
        self.model_path_var = tk.StringVar(value=self.model_path)
        model_path_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=20)
        model_path_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        model_browse_btn = ttk.Button(model_frame, text="Browse...", command=self.browse_model_file)
        model_browse_btn.pack(side=tk.LEFT, padx=5)
        
        # Load model button
        load_model_btn = ttk.Button(settings_frame, text="Load Model", command=self.load_model)
        load_model_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # About frame
        about_frame = ttk.LabelFrame(self.settings_tab, text="About")
        about_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        about_text = tk.Text(about_frame, height=10, width=30, wrap=tk.WORD)
        about_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        about_text.insert(tk.END, "Facial Recognition Application\n\n"
                         "This application uses computer vision to detect and recognize faces "
                         "through your webcam.\n\n"
                         "Built with Python, OpenCV, and face_recognition library.")
        about_text.config(state=tk.DISABLED)
    
    def toggle_camera(self):
        if self.is_running:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        self.camera_index = self.camera_var.get()
        self.recognition_threshold = self.threshold_var.get()
        self.scale_factor = self.scale_factor_var.get()
        self.performance_mode = self.performance_mode_var.get()
        
        # Initialize the video capture
        self.video_capture = cv2.VideoCapture(self.camera_index)
        
        if not self.video_capture.isOpened():
            if self.is_macos:
                self.status_var.set(f"Error: Could not open camera {self.camera_index}. Please check camera permissions in System Preferences.")
                messagebox.showerror("Camera Error", 
                                    "Could not access the camera. On macOS, you need to:\n\n"
                                    "1. Go to System Preferences > Security & Privacy > Privacy > Camera\n"
                                    "2. Ensure Python/Terminal has permission to access the camera\n"
                                    "3. Restart the application after granting permissions")
            else:
                self.status_var.set(f"Error: Could not open camera {self.camera_index}")
            return
        
        # Try to set camera properties for better performance
        if self.performance_mode:
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            # Higher resolution if not in performance mode
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Start the preview thread
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.preview_loop)
        self.thread.daemon = True
        self.thread.start()
        
        # Update the button states
        self.capture_btn.config(state=tk.NORMAL)
        self.is_running = True
        self.status_var.set(f"Camera {self.camera_index} started")
    
    def stop_camera(self):
        # Set the stop event
        if self.stopEvent is not None:
            self.stopEvent.set()
            
            # Wait for the thread to finish, but don't try to join the current thread
            if self.thread is not None and self.thread != threading.current_thread():
                try:
                    self.thread.join(timeout=1.0)  # Add a timeout to prevent hanging
                except RuntimeError:
                    # Ignore "cannot join current thread" error
                    pass
            
            # Release the video capture if it exists
            if self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None
            
            # Update the button states
            self.capture_btn.config(state=tk.DISABLED)
            self.is_running = False
            self.status_var.set("Camera stopped")
            
            # Clear the canvas
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="Camera stopped",
                fill="white",
                font=("Arial", 14)
            )
    
    def preview_loop(self):
        """
        Display a live preview from the camera without facial recognition.
        This is more efficient than continuously analyzing each frame.
        """
        # For macOS, set environment variable to skip authorization request
        if self.is_macos:
            os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"
        
        # Variables for FPS calculation
        frame_count = 0
        fps = 0
        fps_start_time = time.time()
        last_update_time = 0
        min_update_interval = 1.0 / 30.0  # Target 30 FPS for UI updates
        
        try:
            # Loop until the stop event is set
            while not self.stopEvent.is_set():
                # Read a frame from the webcam
                ret, frame = self.video_capture.read()
                
                if not ret or frame is None or frame.size == 0:
                    self.status_var.set("Error: Failed to grab frame from webcam")
                    break
                
                # Calculate time since last update
                current_time = time.time()
                time_since_last_update = current_time - last_update_time
                
                # Only update the display at our target rate
                if time_since_last_update >= min_update_interval:
                    try:
                        # Calculate FPS
                        frame_count += 1
                        elapsed_time = current_time - fps_start_time
                        
                        # Update FPS every second
                        if elapsed_time > 1:
                            fps = frame_count / elapsed_time
                            frame_count = 0
                            fps_start_time = current_time
                        
                        # Display FPS on the frame
                        preview_frame = frame.copy()
                        cv2.putText(
                            preview_frame,
                            f"FPS: {fps:.1f} (Preview Mode)",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )
                        
                        # Convert the frame to RGB for display
                        rgb_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                        
                        # Get canvas dimensions
                        canvas_width = self.canvas.winfo_width()
                        canvas_height = self.canvas.winfo_height()
                        
                        if canvas_width > 1 and canvas_height > 1:
                            # Calculate the aspect ratio
                            img_height, img_width = rgb_frame.shape[:2]
                            aspect_ratio = img_width / img_height
                            
                            # Calculate new dimensions to fit the canvas while maintaining aspect ratio
                            if canvas_width / canvas_height > aspect_ratio:
                                new_height = canvas_height
                                new_width = int(new_height * aspect_ratio)
                            else:
                                new_width = canvas_width
                                new_height = int(new_width / aspect_ratio)
                            
                            # Resize the frame
                            resized_frame = cv2.resize(rgb_frame, (new_width, new_height))
                            
                            # Display the frame using the appropriate method
                            if self.use_pil:
                                try:
                                    # Try using PIL/Pillow
                                    pil_image = PIL.Image.fromarray(resized_frame)
                                    self.photo = PIL.ImageTk.PhotoImage(image=pil_image)
                                    self.canvas.delete("all")  # Clear previous frame
                                    self.canvas.create_image(
                                        canvas_width // 2, 
                                        canvas_height // 2, 
                                        image=self.photo, 
                                        anchor=tk.CENTER
                                    )
                                except (TypeError, ImportError, tk.TclError) as e:
                                    # If PIL fails, switch to fallback method
                                    print(f"PIL display error: {e}. Switching to fallback method.")
                                    self.use_pil = False
                                    self.status_var.set("PIL display error. Using fallback method.")
                            
                            if not self.use_pil:
                                # Fallback method: Convert to Tkinter-compatible format
                                self.display_frame_fallback(resized_frame, canvas_width, canvas_height)
                        
                        # Update the last update time
                        last_update_time = current_time
                    
                    except Exception as e:
                        print(f"Error processing preview frame: {str(e)}")
                        traceback.print_exc()
                
                # Sleep for a short time to reduce CPU usage
                time.sleep(0.01)
        
        except Exception as e:
            self.status_var.set(f"Error in preview loop: {str(e)}")
            print(f"Error in preview loop: {str(e)}")
            traceback.print_exc()
    
    def single_capture_and_analyze(self):
        """
        Single-step process to start camera, capture an image, analyze it, and stop the camera.
        """
        try:
            # Update status
            self.status_var.set("Starting camera...")
            self.window.update()
            
            # Get camera settings
            self.camera_index = self.camera_var.get()
            self.recognition_threshold = self.threshold_var.get()
            self.scale_factor = self.scale_factor_var.get()
            self.performance_mode = self.performance_mode_var.get()  # Get current performance mode setting
            
            # Initialize the video capture with high resolution
            self.video_capture = cv2.VideoCapture(self.camera_index)
            
            if not self.video_capture.isOpened():
                if self.is_macos:
                    self.status_var.set(f"Error: Could not open camera {self.camera_index}. Please check camera permissions in System Preferences.")
                    messagebox.showerror("Camera Error", 
                                        "Could not access the camera. On macOS, you need to:\n\n"
                                        "1. Go to System Preferences > Security & Privacy > Privacy > Camera\n"
                                        "2. Ensure Python/Terminal has permission to access the camera\n"
                                        "3. Restart the application after granting permissions")
                else:
                    self.status_var.set(f"Error: Could not open camera {self.camera_index}")
                    messagebox.showerror("Error", f"Could not open camera {self.camera_index}")
                return
            
            # Try to set the highest resolution possible
            # First try 4K
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
            
            # Check if the resolution was accepted
            actual_width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            # If not, try Full HD
            if actual_width < 3000 or actual_height < 1500:
                self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                
                # Check if Full HD was accepted
                actual_width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                # If not, try HD
                if actual_width < 1900 or actual_height < 1000:
                    self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Update status with actual resolution
            actual_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.status_var.set(f"Capturing image at {actual_width}x{actual_height}...")
            self.window.update()
            
            # Allow camera to warm up
            warmup_frames = 10  # Increased from 5 to 10 for better camera adjustment
            for i in range(warmup_frames):
                self.video_capture.read()
                # Update progress in status bar
                self.status_var.set(f"Warming up camera ({i+1}/{warmup_frames})...")
                self.window.update()
                time.sleep(0.1)
            
            # Capture the image
            ret, frame = self.video_capture.read()
            
            # Release the camera immediately
            self.video_capture.release()
            self.video_capture = None
            
            if not ret or frame is None or frame.size == 0:
                self.status_var.set("Error: Failed to capture image")
                messagebox.showerror("Error", "Failed to capture image from webcam.")
                return
            
            # Update status
            self.status_var.set("Analyzing image...")
            self.window.update()
            
            # Process the frame with facial recognition
            processed_result = detect_and_display_faces(
                frame, 
                self.known_face_encodings, 
                self.known_face_names,
                self.recognition_threshold,
                self.scale_factor
            )
            
            # Get the processed frame from the result
            processed_frame = processed_result.frame
            
            # Convert the frame to RGB for display
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # Calculate the aspect ratio
                img_height, img_width = rgb_frame.shape[:2]
                aspect_ratio = img_width / img_height
                
                # Calculate new dimensions to fit the canvas while maintaining aspect ratio
                if canvas_width / canvas_height > aspect_ratio:
                    new_height = canvas_height
                    new_width = int(new_height * aspect_ratio)
                else:
                    new_width = canvas_width
                    new_height = int(new_width / aspect_ratio)
                
                # Resize the frame using high-quality interpolation
                # Use INTER_CUBIC for upscaling and INTER_AREA for downscaling
                if new_width > img_width or new_height > img_height:
                    # Upscaling - use INTER_CUBIC
                    resized_frame = cv2.resize(rgb_frame, (new_width, new_height), 
                                              interpolation=cv2.INTER_CUBIC)
                else:
                    # Downscaling - use INTER_AREA
                    resized_frame = cv2.resize(rgb_frame, (new_width, new_height), 
                                              interpolation=cv2.INTER_AREA)
                
                # Display the frame using the appropriate method
                if self.use_pil:
                    try:
                        # Try using PIL/Pillow
                        pil_image = PIL.Image.fromarray(resized_frame)
                        self.photo = PIL.ImageTk.PhotoImage(image=pil_image)
                        self.canvas.delete("all")  # Clear previous frame
                        self.canvas.create_image(
                            canvas_width // 2, 
                            canvas_height // 2, 
                            image=self.photo, 
                            anchor=tk.CENTER
                        )
                    except (TypeError, ImportError, tk.TclError) as e:
                        # If PIL fails, switch to fallback method
                        print(f"PIL display error: {e}. Using fallback method.")
                        self.use_pil = False
                        self.status_var.set("PIL display error. Using fallback method.")
                
                if not self.use_pil:
                    # Fallback method: Convert to Tkinter-compatible format
                    self.display_frame_fallback(resized_frame, canvas_width, canvas_height)
            
            # Update the recognition status
            self.update_recognition_status(processed_result)
            
            # Update status with resolution info
            self.status_var.set(f"Analysis complete - Image: {img_width}x{img_height}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture and analyze image: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            traceback.print_exc()
            
            # Make sure to release the camera if an error occurs
            if self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None
    
    def display_frame_fallback(self, frame, canvas_width, canvas_height):
        """
        Fallback method to display frames when PIL/Pillow has issues.
        Uses a more efficient approach with fewer canvas operations.
        """
        try:
            # Clear the canvas
            self.canvas.delete("all")
            
            # Convert the NumPy array to a format Tkinter can use
            height, width = frame.shape[:2]
            
            # Calculate the scaled dimensions to fit the canvas
            if canvas_width / canvas_height > width / height:
                # Canvas is wider than the image
                display_height = min(height, canvas_height)
                display_width = int(display_height * width / height)
            else:
                # Canvas is taller than the image
                display_width = min(width, canvas_width)
                display_height = int(display_width * height / width)
            
            # Resize the image to fit the canvas
            resized = cv2.resize(frame, (display_width, display_height), 
                                interpolation=cv2.INTER_AREA)
            
            # Calculate the position to center the image
            x_offset = (canvas_width - display_width) // 2
            y_offset = (canvas_height - display_height) // 2
            
            # Determine grid size based on performance mode and image size
            if hasattr(self, 'performance_mode') and self.performance_mode:
                # Coarser grid for performance
                grid_size = min(display_width, display_height) // 30
            else:
                # Finer grid for quality
                grid_size = min(display_width, display_height) // 60
            
            # Ensure grid_size is at least 1
            grid_size = max(1, grid_size)
            
            # Create a downsampled version for display
            display_width_grid = (display_width + grid_size - 1) // grid_size
            display_height_grid = (display_height + grid_size - 1) // grid_size
            
            # Create a background rectangle
            self.canvas.create_rectangle(
                x_offset, y_offset,
                x_offset + display_width, y_offset + display_height,
                fill="black", outline=""
            )
            
            # Draw a grid of rectangles
            for y_grid in range(display_height_grid):
                y_start = y_grid * grid_size
                y_end = min((y_grid + 1) * grid_size, display_height)
                
                for x_grid in range(display_width_grid):
                    x_start = x_grid * grid_size
                    x_end = min((x_grid + 1) * grid_size, display_width)
                    
                    # Calculate the average color in this grid cell
                    if y_end > y_start and x_end > x_start:
                        cell = resized[y_start:y_end, x_start:x_end]
                        avg_color = np.mean(cell, axis=(0, 1)).astype(int)
                        r, g, b = avg_color
                        color = f'#{r:02x}{g:02x}{b:02x}'
                        
                        # Draw a rectangle for this cell
                        self.canvas.create_rectangle(
                            x_offset + x_start, 
                            y_offset + y_start,
                            x_offset + x_end, 
                            y_offset + y_end,
                            fill=color, outline=""
                        )
            
            # Add a subtle border to make the image stand out
            self.canvas.create_rectangle(
                x_offset, y_offset,
                x_offset + display_width, y_offset + display_height,
                outline="#333333", width=1
            )
            
            # Add a small indicator about fallback mode
            self.canvas.create_text(
                canvas_width // 2,
                y_offset - 10 if y_offset > 20 else y_offset + display_height + 15,
                text="Fallback display mode",
                fill="#888888",
                font=("Arial", 8)
            )
            
        except Exception as e:
            print(f"Error in fallback display: {str(e)}")
            traceback.print_exc()
            
            # Last resort: just show an error message
            self.canvas.delete("all")
            self.canvas.create_text(
                canvas_width // 2,
                canvas_height // 2,
                text=f"Display Error: {str(e)}",
                fill="red",
                font=("Arial", 12)
            )
    
    def update_recognition_status(self, processed_result):
        """
        Update the recognition status text based on the analyzed frame.
        
        Args:
            processed_result (ProcessedFrame): Object containing the processed frame and face information.
        """
        # Get face information from the processed result
        face_locations = processed_result.face_locations
        face_names = processed_result.face_names
        face_confidences = processed_result.face_confidences
        
        # Count recognized and unknown faces
        recognized_count = sum(1 for name in face_names if name != "Unknown")
        unknown_count = sum(1 for name in face_names if name == "Unknown")
        
        # Create status text
        if len(face_names) == 0:
            status_text = "No faces detected in the image."
        else:
            status_text = f"Analysis complete: {len(face_names)} face(s) detected.\n"
            status_text += f"• {recognized_count} recognized face(s)\n"
            status_text += f"• {unknown_count} unknown face(s)\n\n"
            
            # Add details for each face
            for i, (name, confidence) in enumerate(zip(face_names, face_confidences)):
                if name != "Unknown":
                    status_text += f"Person {i+1}: {name} (Confidence: {confidence:.2f})\n"
                else:
                    status_text += f"Person {i+1}: Unknown\n"
        
        # Update the recognition status text
        self.recognition_status.config(state=tk.NORMAL)
        self.recognition_status.delete(1.0, tk.END)
        self.recognition_status.insert(tk.END, status_text)
        self.recognition_status.config(state=tk.DISABLED)
    
    def browse_training_dir(self):
        directory = filedialog.askdirectory(initialdir=self.training_dir)
        if directory:
            self.training_dir = directory
            self.training_dir_var.set(directory)
    
    def browse_model_file(self):
        file_path = filedialog.askopenfilename(
            initialdir=os.path.dirname(self.model_path),
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if file_path:
            self.model_path = file_path
            self.model_path_var.set(file_path)
    
    def load_model(self):
        model_path = self.model_path_var.get()
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file '{model_path}' does not exist.")
            return
        
        try:
            self.known_face_encodings, self.known_face_names = load_known_faces(model_path)
            self.model_path = model_path
            
            messagebox.showinfo("Success", f"Loaded {len(self.known_face_encodings)} face encodings for {len(set(self.known_face_names))} people.")
            self.update_status()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
    
    def train_model(self):
        training_dir = self.training_dir_var.get()
        
        if not os.path.exists(training_dir):
            messagebox.showerror("Error", f"Training directory '{training_dir}' does not exist.")
            return
        
        # Enable the text widget for editing
        self.training_status.config(state=tk.NORMAL)
        
        # Clear the text widget
        self.training_status.delete(1.0, tk.END)
        
        # Update the status
        self.status_var.set("Training model...")
        self.training_status.insert(tk.END, f"Loading training data from '{training_dir}'...\n")
        self.window.update()
        
        try:
            # Load training data
            known_face_encodings, known_face_names = load_training_data(training_dir)
            
            # Check if any faces were found
            if not known_face_encodings:
                self.training_status.insert(tk.END, "No valid face encodings were generated. Please check your training images.\n")
                self.status_var.set("Training failed")
                return
            
            # Save known faces
            model_path = self.model_path_var.get()
            save_known_faces(known_face_encodings, known_face_names, model_path)
            
            # Update the model
            self.known_face_encodings = known_face_encodings
            self.known_face_names = known_face_names
            self.model_path = model_path
            
            # Print summary
            self.training_status.insert(tk.END, "\nTraining Summary:\n")
            self.training_status.insert(tk.END, f"Total people: {len(set(known_face_names))}\n")
            self.training_status.insert(tk.END, f"Total face encodings: {len(known_face_encodings)}\n\n")
            
            # Print breakdown by person
            person_counts = {}
            for name in known_face_names:
                person_counts[name] = person_counts.get(name, 0) + 1
            
            self.training_status.insert(tk.END, "Face encodings per person:\n")
            for name, count in person_counts.items():
                self.training_status.insert(tk.END, f"  {name}: {count}\n")
            
            self.status_var.set("Training completed successfully")
            self.update_status()
            
            messagebox.showinfo("Success", "Model trained successfully.")
        except Exception as e:
            self.training_status.insert(tk.END, f"Error: {e}\n")
            self.status_var.set("Training failed")
            messagebox.showerror("Error", f"Failed to train model: {e}")
        finally:
            # Disable the text widget
            self.training_status.config(state=tk.DISABLED)
    
    def capture_image(self):
        person_name = self.person_name_var.get().strip()
        
        if not person_name:
            messagebox.showerror("Error", "Please enter a name for the person.")
            return
        
        # Check if the camera is running
        if not self.is_running:
            messagebox.showerror("Error", "Please start the camera first.")
            return
        
        # Create the person's directory if it doesn't exist
        person_dir = os.path.join(self.training_dir_var.get(), person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Use the existing video capture if available
        if self.video_capture is not None and self.video_capture.isOpened():
            # Update status
            self.status_var.set("Capturing training image...")
            self.window.update()
            
            # Read a frame
            ret, frame = self.video_capture.read()
            
            if not ret:
                messagebox.showerror("Error", "Failed to capture image from webcam.")
                return
            
            # Generate a filename based on the current timestamp
            timestamp = int(time.time())
            filename = f"{timestamp}.jpg"
            file_path = os.path.join(person_dir, filename)
            
            # Save the image
            cv2.imwrite(file_path, frame)
            
            # Display confirmation with face detection
            try:
                # Convert the frame to RGB for face detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces in the captured image
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if len(face_locations) == 0:
                    messagebox.showwarning("Warning", 
                                          f"No faces detected in the captured image. The image was saved to {file_path}, "
                                          f"but it may not be useful for training.")
                elif len(face_locations) > 1:
                    messagebox.showwarning("Warning", 
                                          f"Multiple faces ({len(face_locations)}) detected in the captured image. "
                                          f"The image was saved to {file_path}, but it may cause confusion during training.")
                else:
                    messagebox.showinfo("Success", 
                                       f"Image captured with 1 face and saved to {file_path}.")
                    
                    # Update the training status
                    self.training_status.config(state=tk.NORMAL)
                    self.training_status.insert(tk.END, f"Captured image for {person_name}: {filename}\n")
                    self.training_status.see(tk.END)
                    self.training_status.config(state=tk.DISABLED)
            
            except Exception as e:
                # If face detection fails, just show a simple success message
                messagebox.showinfo("Success", f"Image captured and saved to {file_path}.")
                print(f"Error during face detection in captured image: {e}")
        else:
            messagebox.showerror("Error", "Camera is not properly initialized.")
    
    def update_status(self):
        # Update the status bar with the current model info
        if self.known_face_encodings:
            self.status_var.set(f"Model: {self.model_path} | {len(self.known_face_encodings)} face encodings | {len(set(self.known_face_names))} people")
        else:
            self.status_var.set("No model loaded")
    
    def on_close(self):
        # Stop the camera if it's running
        if self.is_running:
            self.stop_camera()
        
        # Destroy the window
        self.window.destroy()

def main():
    # Create the main window
    root = tk.Tk()
    
    # Create the application
    app = FacialRecognitionApp(root, "Facial Recognition Application")
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    # Import face_recognition here to avoid circular imports
    import face_recognition
    main() 