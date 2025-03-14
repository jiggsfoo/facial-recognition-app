package com.example.facialrecognition;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.material.textfield.TextInputEditText;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class TrainingActivity extends AppCompatActivity {
    private static final String TAG = "TrainingActivity";
    private static final int MAX_TRAINING_IMAGES = 5;
    
    // UI components
    private PreviewView viewFinder;
    private TextInputEditText editTextName;
    private Button buttonCapture;
    private Button buttonSave;
    private Button buttonCancel;
    private TextView textInstructions;
    private ImageView[] sampleImageViews;
    
    // Camera variables
    private ImageCapture imageCapture;
    private Camera camera;
    private ProcessCameraProvider cameraProvider;
    private final Executor executor = Executors.newSingleThreadExecutor();
    
    // Face recognition helper
    private FaceRecognitionHelper faceRecognitionHelper;
    
    // Face detector
    private FaceDetector faceDetector;
    
    // Training data
    private List<Bitmap> trainingImages = new ArrayList<>();
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_training);
        
        // Initialize UI components
        viewFinder = findViewById(R.id.viewFinder);
        editTextName = findViewById(R.id.editTextName);
        buttonCapture = findViewById(R.id.buttonCapture);
        buttonSave = findViewById(R.id.buttonSave);
        buttonCancel = findViewById(R.id.buttonCancel);
        textInstructions = findViewById(R.id.textInstructions);
        
        // Initialize sample image views
        sampleImageViews = new ImageView[MAX_TRAINING_IMAGES];
        sampleImageViews[0] = findViewById(R.id.imageSample1);
        sampleImageViews[1] = findViewById(R.id.imageSample2);
        sampleImageViews[2] = findViewById(R.id.imageSample3);
        sampleImageViews[3] = findViewById(R.id.imageSample4);
        sampleImageViews[4] = findViewById(R.id.imageSample5);
        
        // Initialize face recognition helper
        faceRecognitionHelper = new FaceRecognitionHelper(this);
        
        // Initialize face detector
        FaceDetectorOptions options = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                .setMinFaceSize(0.15f)
                .build();
        
        faceDetector = FaceDetection.getClient(options);
        
        // Set up UI listeners
        setupUIListeners();
        
        // Start camera
        startCamera();
    }
    
    private void setupUIListeners() {
        // Capture button
        buttonCapture.setOnClickListener(v -> captureTrainingImage());
        
        // Save button
        buttonSave.setOnClickListener(v -> saveTrainingData());
        
        // Cancel button
        buttonCancel.setOnClickListener(v -> finish());
    }
    
    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                
                // Set up the preview use case
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(viewFinder.getSurfaceProvider());
                
                // Set up the image capture use case
                imageCapture = new ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                        .build();
                
                // Select front camera for training
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                        .build();
                
                // Unbind any bound use cases before rebinding
                cameraProvider.unbindAll();
                
                // Bind use cases to camera
                camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture);
                
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Error starting camera: " + e.getMessage());
            }
        }, ContextCompat.getMainExecutor(this));
    }
    
    private void captureTrainingImage() {
        if (imageCapture == null) {
            Toast.makeText(this, "Camera not initialized", Toast.LENGTH_SHORT).show();
            return;
        }
        
        // Check if we already have the maximum number of training images
        if (trainingImages.size() >= MAX_TRAINING_IMAGES) {
            Toast.makeText(this, "Maximum number of training images reached", Toast.LENGTH_SHORT).show();
            return;
        }
        
        // Create temporary file for the image
        File outputDir = getApplicationContext().getCacheDir();
        File outputFile;
        try {
            String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(System.currentTimeMillis());
            outputFile = File.createTempFile(timeStamp, ".jpg", outputDir);
        } catch (Exception e) {
            Log.e(TAG, "Error creating temporary file: " + e.getMessage());
            Toast.makeText(this, "Error creating temporary file", Toast.LENGTH_SHORT).show();
            return;
        }
        
        // Create output options object
        ImageCapture.OutputFileOptions outputFileOptions = new ImageCapture.OutputFileOptions.Builder(outputFile).build();
        
        // Disable the capture button while capturing
        buttonCapture.setEnabled(false);
        
        // Capture the image
        imageCapture.takePicture(outputFileOptions, executor, new ImageCapture.OnImageSavedCallback() {
            @Override
            public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                // Load the captured image
                Bitmap bitmap = BitmapFactory.decodeFile(outputFile.getAbsolutePath());
                
                // Detect faces in the image
                InputImage inputImage = InputImage.fromBitmap(bitmap, 0);
                faceDetector.process(inputImage)
                        .addOnSuccessListener(faces -> {
                            if (faces.isEmpty()) {
                                runOnUiThread(() -> {
                                    Toast.makeText(TrainingActivity.this, R.string.no_face_detected, Toast.LENGTH_SHORT).show();
                                    buttonCapture.setEnabled(true);
                                });
                                return;
                            }
                            
                            if (faces.size() > 1) {
                                runOnUiThread(() -> {
                                    Toast.makeText(TrainingActivity.this, R.string.multiple_faces_detected, Toast.LENGTH_SHORT).show();
                                    buttonCapture.setEnabled(true);
                                });
                                return;
                            }
                            
                            // Extract the face from the image
                            Face face = faces.get(0);
                            try {
                                // Get the face bounding box
                                Bitmap faceBitmap = extractFace(bitmap, face);
                                
                                // Add to training images
                                trainingImages.add(faceBitmap);
                                
                                // Update UI
                                runOnUiThread(() -> {
                                    // Show the captured face in the sample image view
                                    int index = trainingImages.size() - 1;
                                    sampleImageViews[index].setImageBitmap(faceBitmap);
                                    
                                    // Enable the save button if we have at least one image
                                    buttonSave.setEnabled(true);
                                    
                                    // Update instructions
                                    if (trainingImages.size() >= MAX_TRAINING_IMAGES) {
                                        textInstructions.setText("Maximum number of training images reached. You can save now.");
                                        buttonCapture.setEnabled(false);
                                    } else {
                                        textInstructions.setText("Captured " + trainingImages.size() + " of " + MAX_TRAINING_IMAGES + " images. Please capture more from different angles.");
                                        buttonCapture.setEnabled(true);
                                    }
                                });
                                
                            } catch (Exception e) {
                                Log.e(TAG, "Error extracting face: " + e.getMessage());
                                runOnUiThread(() -> {
                                    Toast.makeText(TrainingActivity.this, "Error extracting face", Toast.LENGTH_SHORT).show();
                                    buttonCapture.setEnabled(true);
                                });
                            }
                        })
                        .addOnFailureListener(e -> {
                            Log.e(TAG, "Face detection failed: " + e.getMessage());
                            runOnUiThread(() -> {
                                Toast.makeText(TrainingActivity.this, "Face detection failed", Toast.LENGTH_SHORT).show();
                                buttonCapture.setEnabled(true);
                            });
                        });
            }
            
            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                Log.e(TAG, "Error capturing image: " + exception.getMessage());
                runOnUiThread(() -> {
                    Toast.makeText(TrainingActivity.this, "Error capturing image", Toast.LENGTH_SHORT).show();
                    buttonCapture.setEnabled(true);
                });
            }
        });
    }
    
    private Bitmap extractFace(Bitmap image, Face face) {
        // Get the face bounding box
        android.graphics.Rect boundingBox = face.getBoundingBox();
        
        // Ensure the bounding box is within the image bounds
        boundingBox.left = Math.max(0, boundingBox.left);
        boundingBox.top = Math.max(0, boundingBox.top);
        boundingBox.right = Math.min(image.getWidth(), boundingBox.right);
        boundingBox.bottom = Math.min(image.getHeight(), boundingBox.bottom);
        
        // Extract the face region
        return Bitmap.createBitmap(
                image,
                boundingBox.left,
                boundingBox.top,
                boundingBox.width(),
                boundingBox.height()
        );
    }
    
    private void saveTrainingData() {
        // Check if we have any training images
        if (trainingImages.isEmpty()) {
            Toast.makeText(this, "No training images captured", Toast.LENGTH_SHORT).show();
            return;
        }
        
        // Get the person's name
        String name = editTextName.getText().toString().trim();
        if (TextUtils.isEmpty(name)) {
            Toast.makeText(this, "Please enter a name", Toast.LENGTH_SHORT).show();
            return;
        }
        
        // Disable UI while saving
        buttonCapture.setEnabled(false);
        buttonSave.setEnabled(false);
        buttonCancel.setEnabled(false);
        textInstructions.setText("Training in progress...");
        
        // Add the person to the face recognition helper
        faceRecognitionHelper.addPerson(name, trainingImages, new FaceRecognitionHelper.TrainingListener() {
            @Override
            public void onTrainingSuccess() {
                runOnUiThread(() -> {
                    Toast.makeText(TrainingActivity.this, R.string.training_complete, Toast.LENGTH_SHORT).show();
                    finish();
                });
            }
            
            @Override
            public void onTrainingFailure(String error) {
                runOnUiThread(() -> {
                    Toast.makeText(TrainingActivity.this, error, Toast.LENGTH_SHORT).show();
                    
                    // Re-enable UI
                    buttonCapture.setEnabled(trainingImages.size() < MAX_TRAINING_IMAGES);
                    buttonSave.setEnabled(true);
                    buttonCancel.setEnabled(true);
                    textInstructions.setText(R.string.training_instructions);
                });
            }
        });
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        
        // Clean up resources
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }
        
        if (faceDetector != null) {
            faceDetector.close();
        }
    }
} 