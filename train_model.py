#!/usr/bin/env python3
"""
Training script for facial recognition.
This script processes training images and generates face encodings.
"""

import os
import argparse
from utils import load_training_data, save_known_faces

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train facial recognition model')
    parser.add_argument('--training-dir', type=str, default='training_data',
                        help='Directory containing training images')
    parser.add_argument('--output', type=str, default='known_faces.pkl',
                        help='Output file to save face encodings')
    args = parser.parse_args()
    
    # Check if training directory exists
    if not os.path.exists(args.training_dir):
        print(f"Error: Training directory '{args.training_dir}' does not exist.")
        print("Please create the directory and add training images.")
        return
    
    # Load training data
    print(f"Loading training data from '{args.training_dir}'...")
    known_face_encodings, known_face_names = load_training_data(args.training_dir)
    
    # Check if any faces were found
    if not known_face_encodings:
        print("No valid face encodings were generated. Please check your training images.")
        return
    
    # Save known faces
    save_known_faces(known_face_encodings, known_face_names, args.output)
    
    # Print summary
    print("\nTraining Summary:")
    print(f"Total people: {len(set(known_face_names))}")
    print(f"Total face encodings: {len(known_face_encodings)}")
    
    # Print breakdown by person
    person_counts = {}
    for name in known_face_names:
        person_counts[name] = person_counts.get(name, 0) + 1
    
    print("\nFace encodings per person:")
    for name, count in person_counts.items():
        print(f"  {name}: {count}")

if __name__ == "__main__":
    main() 