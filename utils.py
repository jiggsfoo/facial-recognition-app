import os
import pickle
import face_recognition
import cv2
import numpy as np
from PIL import Image
import traceback

class ProcessedFrame:
    """
    A class to hold a processed frame and its face information.
    """
    def __init__(self, frame, face_locations=None, face_names=None, face_confidences=None):
        self.frame = frame
        self.face_locations = face_locations or []
        self.face_names = face_names or []
        self.face_confidences = face_confidences or []

def load_training_data(training_dir="training_data"):
    """
    Load training images and generate face encodings.
    
    Args:
        training_dir (str): Path to the directory containing training images.
        
    Returns:
        tuple: (known_face_encodings, known_face_names)
    """
    known_face_encodings = []
    known_face_names = []
    
    # Iterate through each person's directory
    for person_name in os.listdir(training_dir):
        person_dir = os.path.join(training_dir, person_name)
        
        # Skip if not a directory
        if not os.path.isdir(person_dir):
            continue
        
        # Process each image in the person's directory
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            
            # Skip if not an image file
            if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            try:
                # Load the image
                image = face_recognition.load_image_file(image_path)
                
                # Find face locations in the image
                face_locations = face_recognition.face_locations(image)
                
                # Skip if no faces or multiple faces are detected
                if len(face_locations) != 1:
                    print(f"Warning: {image_path} contains {len(face_locations)} faces. Skipping.")
                    continue
                
                # Generate face encoding
                face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                
                # Add to known faces
                known_face_encodings.append(face_encoding)
                known_face_names.append(person_name)
                
                print(f"Processed: {image_path}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    
    return known_face_encodings, known_face_names

def save_known_faces(known_face_encodings, known_face_names, filename="known_faces.pkl"):
    """
    Save known face encodings and names to a file.
    
    Args:
        known_face_encodings (list): List of face encodings.
        known_face_names (list): List of corresponding names.
        filename (str): Path to save the data.
    """
    data = {
        "encodings": known_face_encodings,
        "names": known_face_names
    }
    
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Saved {len(known_face_encodings)} face encodings to {filename}")

def load_known_faces(filename="known_faces.pkl"):
    """
    Load known face encodings and names from a file.
    
    Args:
        filename (str): Path to the saved data.
        
    Returns:
        tuple: (known_face_encodings, known_face_names)
    """
    if not os.path.exists(filename):
        return [], []
    
    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data['encodings'])} face encodings from {filename}")
    return data["encodings"], data["names"]

def detect_and_display_faces(frame, known_face_encodings=None, known_face_names=None, recognition_threshold=0.6, scale_factor=0.5):
    """
    Detect faces in a frame and optionally recognize them.
    
    Args:
        frame (numpy.ndarray): The frame to process.
        known_face_encodings (list, optional): List of known face encodings.
        known_face_names (list, optional): List of corresponding names.
        recognition_threshold (float): Threshold for face recognition (lower is stricter).
        scale_factor (float): Factor to scale down the image for faster processing (0.5 = half size).
        
    Returns:
        ProcessedFrame: An object containing the processed frame and face information.
    """
    try:
        # Validate input frame
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            print("Error: Invalid frame provided to detect_and_display_faces")
            # Return a blank frame with error message
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "Error: Invalid frame", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return ProcessedFrame(blank_frame)
        
        # Make a copy of the frame to avoid modifying the original
        processed_frame = frame.copy()
        
        # Initialize lists to store face information
        face_locations = []
        face_names = []
        face_confidences = []
        
        try:
            # Convert the frame from BGR (OpenCV) to RGB (face_recognition)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Scale down the frame for faster face detection
            if scale_factor < 1.0:
                h, w = rgb_frame.shape[:2]
                small_frame = cv2.resize(rgb_frame, (0, 0), fx=scale_factor, fy=scale_factor)
            else:
                small_frame = rgb_frame
            
            # Find all face locations in the smaller frame
            face_locations = face_recognition.face_locations(small_frame)
            
            # Scale the face locations back to the original size
            if scale_factor < 1.0:
                face_locations = [(int(top/scale_factor), int(right/scale_factor), 
                                  int(bottom/scale_factor), int(left/scale_factor)) 
                                 for top, right, bottom, left in face_locations]
            
            # Only attempt recognition if faces were found
            if face_locations:
                # Get face encodings from the original frame using the scaled locations
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                # Loop through each face found in the frame
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    name = "Unknown"
                    confidence = 0.0
                    
                    # Only attempt recognition if we have known faces
                    if known_face_encodings and known_face_names and len(known_face_encodings) > 0:
                        # Compare the face with known faces
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        
                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            distance = face_distances[best_match_index]
                            confidence = 1.0 - distance
                            
                            # If the best match is below the threshold, use the name
                            if distance < recognition_threshold:
                                name = known_face_names[best_match_index]
                    
                    # Store the face information
                    face_names.append(name)
                    face_confidences.append(confidence)
                    
                    # Draw a rectangle around the face
                    cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Draw the name below the face
                    cv2.rectangle(processed_frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(processed_frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                    
                    # Add confidence if recognized
                    if name != "Unknown":
                        confidence_text = f"{confidence:.2f}"
                        cv2.putText(processed_frame, confidence_text, (left + 6, top - 6), 
                                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
        
        except Exception as e:
            print(f"Error in face detection/recognition: {str(e)}")
            traceback.print_exc()
            # Add error message to the frame
            cv2.putText(processed_frame, f"Error: {str(e)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Create and return a ProcessedFrame object
        return ProcessedFrame(processed_frame, face_locations, face_names, face_confidences)
    
    except Exception as e:
        print(f"Critical error in detect_and_display_faces: {str(e)}")
        traceback.print_exc()
        # Return a blank frame with error message
        try:
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, f"Error: {str(e)}", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Return a ProcessedFrame with empty face information
            return ProcessedFrame(blank_frame)
        except:
            # Last resort fallback
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            return ProcessedFrame(blank) 