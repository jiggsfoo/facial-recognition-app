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
import subprocess
import traceback
import numpy as np
from utils import load_known_faces, detect_and_display_faces

# Set up Info.plist for macOS
if platform.system() == 'Darwin':
    # Check if Info.plist exists in the current directory
    if os.path.exists('Info.plist'):
        # Set the environment variable to use the Info.plist file
        os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
        
        # Suppress the AVCaptureDevice warning
        os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'
        
        # Set the NSCameraUsageDescription environment variable
        os.environ['NSCameraUsageDescription'] = 'This application needs access to your camera for facial recognition.'
        
        # Print a message about using Info.plist
        print("Using Info.plist for macOS camera configuration")

def force_camera_permission_request(camera_index=0):
    """
    Force macOS to show the camera permission dialog by explicitly trying to access the camera.
    This helps ensure Terminal/Python appears in the permissions list.
    """
    if platform.system() != 'Darwin':
        return
    
    print("Attempting to trigger camera permission dialog...")
    
    try:
        # Create a visible window with instructions
        cv2.namedWindow('Camera Access Required', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera Access Required', 640, 480)
        
        # Create a blank image with text
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Requesting Camera Access...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, "If no permission dialog appears:", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(img, "1. Open System Settings manually", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(img, "2. Go to Privacy & Security > Camera", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(img, "3. Add Terminal/Python to the list", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(img, "Press any key to continue...", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        # Show the instructions
        cv2.imshow('Camera Access Required', img)
        cv2.waitKey(2000)  # Wait for 2 seconds or key press
        
        # Try multiple approaches to trigger the permission dialog
        
        # Approach 1: Standard OpenCV camera access
        print("Approach 1: Standard OpenCV camera access")
        camera = cv2.VideoCapture(camera_index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Try to read frames
        for i in range(10):
            ret, frame = camera.read()
            if ret and frame is not None:
                cv2.imshow('Camera Access Required', frame)
                cv2.waitKey(100)
                print(f"Frame {i+1} captured")
        
        camera.release()
        
        # Approach 2: Use AVFoundation directly via environment variables
        print("Approach 2: Using AVFoundation environment variables")
        os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '0'  # Force authorization
        os.environ['OPENCV_VIDEOIO_DEBUG'] = '1'  # Enable debug output
        
        camera2 = cv2.VideoCapture(camera_index)
        for i in range(5):
            ret, frame = camera2.read()
            if ret and frame is not None:
                cv2.imshow('Camera Access Required', frame)
                cv2.waitKey(200)
        
        camera2.release()
        
        # Approach 3: Try with different backend
        print("Approach 3: Trying with different backend")
        if hasattr(cv2, 'CAP_AVFOUNDATION'):
            camera3 = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
            for i in range(5):
                ret, frame = camera3.read()
                if ret and frame is not None:
                    cv2.imshow('Camera Access Required', frame)
                    cv2.waitKey(200)
            camera3.release()
        
        # Clean up
        cv2.destroyAllWindows()
        
        # Reset environment variables
        os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'
        
        print("Camera permission request attempts completed.")
        print("If you didn't see a permission dialog, you may need to manually add Terminal/Python")
        print("to the list of applications allowed to access the camera in System Settings.")
        
        # Ask if the user wants to open System Settings
        response = input("\nWould you like to open System Settings to manually add Terminal/Python? (y/n): ")
        if response.lower() in ['y', 'yes']:
            try:
                subprocess.run(['open', 'x-apple.systempreferences:com.apple.preference.security?Privacy_Camera'])
                print("\nIn System Settings:")
                print("1. Click the '+' button under the list of applications")
                print("2. Navigate to /Applications/Utilities and select Terminal")
                print("3. Click 'Add' and ensure the checkbox next to Terminal is checked")
                print("4. Restart this application")
            except Exception as e:
                print(f"Error opening System Settings: {e}")
                try:
                    # Fallback to older method
                    subprocess.run(['open', '/System/Library/PreferencePanes/Security.prefPane'])
                except:
                    pass
    
    except Exception as e:
        print(f"Error during camera permission request: {e}")
        traceback.print_exc()
        # Clean up in case of error
        try:
            cv2.destroyAllWindows()
        except:
            pass

def check_macos_camera_permissions(camera_index=0):
    """
    Check and request camera permissions on macOS.
    Returns True if permissions are granted, False otherwise.
    """
    if platform.system() != 'Darwin':
        return True
    
    print("Checking camera permissions on macOS...")
    
    # Make a more explicit attempt to access the camera to trigger macOS permission dialog
    # This approach forces macOS to register the application in the permissions list
    try:
        # First, try to open the camera with explicit properties to trigger the permission dialog
        test_capture = cv2.VideoCapture(camera_index)
        
        # Set some properties to ensure the camera is actually accessed
        test_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        test_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Try to read a frame to force the permission dialog
        ret, frame = test_capture.read()
        
        # Check if we successfully accessed the camera
        has_permission = test_capture.isOpened() and ret
        
        # Release the camera
        test_capture.release()
        
        if has_permission:
            print("Camera permission granted.")
            return True
    except Exception as e:
        print(f"Error accessing camera: {e}")
        has_permission = False
    
    # If we couldn't access the camera, guide the user
    print("\nCamera Permission Required")
    print("This app needs camera access to function properly.")
    print("\nPlease follow these steps:")
    print("1. Go to System Preferences > Security & Privacy > Privacy > Camera")
    print("2. Ensure Terminal/Python has permission to access the camera")
    print("3. Restart this application after granting permissions")
    
    # Ask if user wants to open System Preferences
    response = input("\nWould you like to open System Preferences now? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        # Open System Preferences to the Camera privacy settings
        try:
            # First try the newer macOS 13+ approach
            subprocess.run([
                'open', 
                'x-apple.systempreferences:com.apple.preference.security?Privacy_Camera'
            ])
            
            print("\nAfter Terminal/Python appears in the list, please:")
            print("1. Check the box next to Terminal/Python to grant permission")
            print("2. Restart this application")
            
            # Give additional instructions for troubleshooting
            print("\nIf Terminal/Python doesn't appear in the list:")
            print("1. Close this application")
            print("2. Run it again to trigger another permission request")
            print("3. If it still doesn't appear, try running the application with 'sudo'")
            print("   or try running it from a different terminal application")
        except Exception as e:
            print(f"Error opening System Preferences: {e}")
            
            # Fallback for older macOS versions
            try:
                subprocess.run(['open', '/System/Library/PreferencePanes/Security.prefPane'])
                print("\nPlease navigate to the Privacy tab, then select Camera from the list.")
            except Exception as e:
                print(f"Error opening Security preferences: {e}")
    
    return False

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
    
    # For macOS, force a camera permission request to ensure Terminal/Python appears in the list
    if is_macos:
        force_camera_permission_request(args.camera)
        
        # Then check if permission was granted
        if not check_macos_camera_permissions(args.camera):
            sys.exit(1)
    
    # Load known faces
    known_face_encodings, known_face_names = load_known_faces(args.model)
    
    if not known_face_encodings:
        print(f"Warning: No face encodings found in {args.model}.")
        print("You may want to train the model first using train_model.py.")
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
            
            # Ask if user wants to open System Preferences
            response = input("\nWould you like to open System Preferences now? (y/n): ")
            
            if response.lower() in ['y', 'yes']:
                # Open System Preferences to the Camera privacy settings
                try:
                    subprocess.run([
                        'open', 
                        'x-apple.systempreferences:com.apple.preference.security?Privacy_Camera'
                    ])
                    print("\nAfter granting camera permissions, please restart this application.")
                except Exception as e:
                    print(f"Error opening System Preferences: {e}")
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