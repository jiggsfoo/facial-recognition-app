#!/usr/bin/env python3
"""
Facial Recognition Application.
This application uses the webcam to detect and recognize faces.
"""

import os
import argparse
import cv2
import time
import platform
import sys
from utils import load_known_faces, detect_and_display_faces

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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Facial Recognition Application')
    parser.add_argument('--model', type=str, default='known_faces.pkl',
                        help='Path to the trained model file')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index (default: 0)')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Face recognition threshold (lower is stricter, default: 0.6)')
    parser.add_argument('--display-fps', action='store_true',
                        help='Display FPS counter')
    args = parser.parse_args()
    
    # Check if running on macOS
    is_macos = platform.system() == 'Darwin'
    
    # For macOS, set environment variable to skip authorization request if not already set
    if is_macos and 'OPENCV_AVFOUNDATION_SKIP_AUTH' not in os.environ:
        os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"
        print("Note: On macOS, you may need to grant camera permissions in System Preferences.")
        print("If the camera doesn't work, please check:")
        print("  1. System Preferences > Security & Privacy > Privacy > Camera")
        print("  2. Ensure Python/Terminal has permission to access the camera")
        print("  3. Restart the application after granting permissions")
    
    # Load known faces
    known_face_encodings, known_face_names = load_known_faces(args.model)
    
    # Check if any faces were loaded
    if not known_face_encodings:
        print(f"Warning: No face encodings loaded from '{args.model}'.")
        print("The application will run in detection-only mode.")
    else:
        print(f"Loaded {len(known_face_encodings)} face encodings for {len(set(known_face_names))} people.")
    
    # Initialize webcam
    print(f"Initializing webcam (device: {args.camera})...")
    video_capture = cv2.VideoCapture(args.camera)
    
    if not video_capture.isOpened():
        print(f"Error: Could not open webcam (device: {args.camera}).")
        if is_macos:
            print("On macOS, this is often due to permission issues.")
            print("Please check your camera permissions in System Preferences > Security & Privacy > Privacy > Camera")
        return
    
    print("Webcam initialized successfully.")
    print("Press 'q' to quit.")
    
    # Variables for FPS calculation
    frame_count = 0
    fps = 0
    fps_start_time = time.time()
    
    try:
        # Main loop
        while True:
            # Read a frame from the webcam
            ret, frame = video_capture.read()
            
            if not ret:
                print("Error: Failed to grab frame from webcam.")
                break
            
            # Process the frame
            processed_frame = detect_and_display_faces(
                frame, 
                known_face_encodings, 
                known_face_names,
                args.threshold
            )
            
            # Calculate and display FPS if enabled
            if args.display_fps:
                frame_count += 1
                elapsed_time = time.time() - fps_start_time
                
                # Update FPS every second
                if elapsed_time > 1:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    fps_start_time = time.time()
                
                # Display FPS on the frame
                cv2.putText(
                    processed_frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            
            # Display the resulting frame
            cv2.imshow('Facial Recognition', processed_frame)
            
            # Check for 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        # Release resources
        video_capture.release()
        cv2.destroyAllWindows()
        print("Application terminated.")

if __name__ == "__main__":
    main() 