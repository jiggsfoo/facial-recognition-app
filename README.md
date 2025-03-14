# Facial Recognition Application

This application uses computer vision to detect and recognize faces through your webcam.

## Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/jiggsfoo/facial-recognition-app.git
cd facial-recognition-app
```

2. Set up the Python environment using mise:
```bash
# If you don't have mise installed yet
# On macOS: brew install mise
# Or follow instructions at https://mise.jdx.dev/getting-started.html

# Create a .mise.toml file (if not already present)
echo '[tools]
python = "3.10"' > .mise.toml

# Install mise and set up the environment
mise install
mise use python

# Create a virtual environment within mise
mise exec python -- -m venv .venv
mise use .venv
```

3. Install Tkinter (required for the GUI):
   - **macOS**: `brew install python-tk`
   - **Ubuntu/Debian**: `sudo apt-get install python3-tk`
   - **Windows**: Tkinter is included with Python installation

4. Install dependencies:
```bash
mise run install
# or
mise exec -- pip install -r requirements.txt
```

Note: Installing `dlib` might require additional build tools. On macOS, you might need to install CMake:
```bash
brew install cmake
```

### Known Issues and Troubleshooting

#### PIL/Tkinter Integration on macOS

On macOS, there may be issues with PIL/Pillow integration with Tkinter, especially in virtual environments. This can cause errors when displaying images in the GUI application. The application includes a fallback display method that will be used automatically if PIL/Tkinter integration fails.

To test if PIL/Tkinter integration is working on your system:
```bash
mise run test-pil
```

If you see an error like `bad argument type for built-in operation`, the application will use the fallback display method, which may be slower or lower quality.

To fix this issue, you can try:
1. Installing Tkinter globally: `brew install python-tk`
2. Using a different Python installation method (e.g., system Python instead of mise)
3. Using the command-line application instead of the GUI

#### macOS Camera Permissions

On macOS, you may need to grant camera permissions to use the webcam:

1. Go to System Preferences > Security & Privacy > Privacy > Camera
2. Ensure that Terminal or Python has permission to access the camera
3. If you've just granted permission, you may need to restart the application

The application includes an Info.plist file for better compatibility with macOS camera features, including Continuity Camera (using your iPhone as a webcam). The application sets the necessary environment variables to handle camera authorization in macOS, but you still need to grant permissions in System Preferences.

If you see warnings about AVCaptureDeviceTypeExternal being deprecated, these are informational and don't affect the functionality of the application.

## Usage

### Training the Model

1. Create a directory structure for training images:
```
training_data/
    person1/
        image1.jpg
        image2.jpg
        ...
    person2/
        image1.jpg
        ...
    ...
```

2. Run the training script:
```bash
mise run train
# or
mise exec -- python train_model.py
```

### Running the Application

#### Command Line Interface

1. Start the facial recognition application:
```bash
mise run run
# or
mise exec -- python facial_recognition.py
```

2. Press 'q' to quit the application.

#### Graphical User Interface

The application includes a graphical user interface (GUI) for easier interaction. The GUI provides the following features:

- **Single-button facial recognition**: Simply click the "Capture & Analyze" button to take a photo and analyze it.
- **Training interface**: Add new faces to the recognition model.
- **Adjustable settings**: Configure recognition threshold, detection scale, and performance mode.
- **Status information**: View detailed recognition results.

### Using the Single-Button Facial Recognition

1. **Launch the application**: Run the GUI application using `mise run gui` or `python gui_app.py`.
2. **Capture and analyze**: Click the "Capture & Analyze" button. This will:
   - Start the camera
   - Capture a high-resolution image (up to 4K if your camera supports it)
   - Analyze the image for faces
   - Display the results
   - Stop the camera automatically

3. **View results**: After analysis, the application will display:
   - The captured image with face boxes and names
   - A detailed summary of recognized and unknown faces
   - Confidence scores for each recognized face

4. **Adjust settings**: You can adjust the following settings before capturing:
   - **Camera**: Select which camera to use
   - **Recognition Threshold**: Lower values are stricter (require closer matches)
   - **Detection Scale**: Lower values improve performance but may reduce accuracy
   - **Performance Mode**: Enable for faster processing on less powerful hardware, disable for higher quality display

This single-button approach is more efficient than continuous analysis, especially on less powerful hardware.

### Training New Faces

1. Click the "Add New Person" button in the GUI to start training a new face.
2. Follow the on-screen instructions to capture multiple images of the new person.
3. The application will automatically train a model based on the captured images.

### macOS Camera Permissions

On macOS, you may need to grant camera permissions to use the webcam:

1. Go to System Preferences > Security & Privacy > Privacy > Camera
2. Ensure that Terminal or Python has permission to access the camera
3. If you've just granted permission, you may need to restart the application

The application includes an Info.plist file for better compatibility with macOS camera features, including Continuity Camera (using your iPhone as a webcam). The application sets the necessary environment variables to handle camera authorization in macOS, but you still need to grant permissions in System Preferences.

If you see warnings about AVCaptureDeviceTypeExternal being deprecated, these are informational and don't affect the functionality of the application.

## Mise Tasks

This project includes predefined mise tasks for common operations:

| Task | Description | Command |
|------|-------------|---------|
| `install` | Install dependencies | `mise run install` |
| `train` | Train the facial recognition model | `mise run train` |
| `run` | Run the command-line application | `mise run run` |
| `gui` | Run the GUI application | `mise run gui` |
| `test-pil` | Test PIL/Tkinter integration | `mise run test-pil` |

## Command Line Options

### train_model.py

```bash
mise run train -- --training-dir PATH_TO_TRAINING_DATA --output OUTPUT_FILE
# or
mise exec -- python train_model.py --training-dir PATH_TO_TRAINING_DATA --output OUTPUT_FILE
```

Options:
- `--training-dir`: Directory containing training images (default: "training_data")
- `--output`: Output file to save face encodings (default: "known_faces.pkl")

### facial_recognition.py

```bash
mise run run -- --model MODEL_FILE --camera CAMERA_INDEX --threshold THRESHOLD --display-fps
# or
mise exec -- python facial_recognition.py --model MODEL_FILE --camera CAMERA_INDEX --threshold THRESHOLD --display-fps
```

Options:
- `--model`: Path to the trained model file (default: "known_faces.pkl")
- `--camera`: Camera device index (default: 0)
- `--threshold`: Face recognition threshold (lower is stricter, default: 0.6)
- `--display-fps`: Display FPS counter

## Project Structure

- `facial_recognition.py`: Command-line application for webcam-based facial recognition
- `gui_app.py`: Graphical user interface for the facial recognition system
- `train_model.py`: Script to train the facial recognition model
- `utils.py`: Utility functions for face detection and recognition
- `training_data/`: Directory containing training images
- `known_faces.pkl`: Saved face encodings (generated after training)
- `.mise.toml`: Configuration file for mise environment and tasks
- `Info.plist`: macOS property list file for camera permissions and Continuity Camera support
- `test_pil_tk.py`: Test script for PIL/Tkinter integration

## License

MIT 