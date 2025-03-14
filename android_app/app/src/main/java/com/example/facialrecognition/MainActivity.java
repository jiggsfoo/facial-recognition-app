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
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.face.Face;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final int REQUEST_CODE_PERMISSIONS = 10;
    private static final String[] REQUIRED_PERMISSIONS = new String[]{Manifest.permission.CAMERA};
    
    // UI components
    private PreviewView viewFinder;
    private ImageView imageResult;
    private Button buttonCaptureAnalyze;
    private Button buttonAddPerson;
    private Button buttonSettings;
    private LinearLayout settingsPanel;
    private SeekBar seekBarThreshold;
    private SeekBar seekBarScale;
    private CheckBox checkBoxPerformanceMode;
    private View resultPanel;
    private TextView textResults;
    
    // Camera variables
    private ImageCapture imageCapture;
    private Camera camera;
    private ProcessCameraProvider cameraProvider;
    private final Executor executor = Executors.newSingleThreadExecutor();
    
    // Face recognition helper
    private FaceRecognitionHelper faceRecognitionHelper;
    
    // Settings
    private float recognitionThreshold = 0.6f;
    private float detectionScale = 0.5f;
    private boolean performanceMode = true;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        try {
            // Initialize UI components
            viewFinder = findViewById(R.id.viewFinder);
            imageResult = findViewById(R.id.imageResult);
            buttonCaptureAnalyze = findViewById(R.id.buttonCaptureAnalyze);
            buttonAddPerson = findViewById(R.id.buttonAddPerson);
            buttonSettings = findViewById(R.id.buttonSettings);
            settingsPanel = findViewById(R.id.settingsPanel);
            seekBarThreshold = findViewById(R.id.seekBarThreshold);
            seekBarScale = findViewById(R.id.seekBarScale);
            checkBoxPerformanceMode = findViewById(R.id.checkBoxPerformanceMode);
            resultPanel = findViewById(R.id.resultPanel);
            textResults = findViewById(R.id.textResults);
            
            // Initialize face recognition helper with error handling
            try {
                faceRecognitionHelper = new FaceRecognitionHelper(this);
                Log.d(TAG, "FaceRecognitionHelper initialized successfully");
            } catch (Exception e) {
                Log.e(TAG, "Error initializing FaceRecognitionHelper: " + e.getMessage());
                Toast.makeText(this, "Face recognition features may be limited", Toast.LENGTH_LONG).show();
                // Continue with a null helper - we'll check for null before using it
            }
            
            // Set up UI listeners
            setupUIListeners();
            
            // Request camera permissions
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
            }
        } catch (Exception e) {
            Log.e(TAG, "Error in onCreate: " + e.getMessage());
            Toast.makeText(this, "Error initializing app: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }
    
    private void setupUIListeners() {
        // Capture and analyze button
        buttonCaptureAnalyze.setOnClickListener(v -> captureAndAnalyze());
        
        // Add person button
        buttonAddPerson.setOnClickListener(v -> {
            Intent intent = new Intent(MainActivity.this, TrainingActivity.class);
            startActivity(intent);
        });
        
        // Settings button
        buttonSettings.setOnClickListener(v -> {
            if (settingsPanel.getVisibility() == View.VISIBLE) {
                settingsPanel.setVisibility(View.GONE);
            } else {
                settingsPanel.setVisibility(View.VISIBLE);
            }
        });
        
        // Recognition threshold seek bar
        seekBarThreshold.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                recognitionThreshold = progress / 100.0f;
                faceRecognitionHelper.setRecognitionThreshold(recognitionThreshold);
            }
            
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        
        // Detection scale seek bar
        seekBarScale.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                detectionScale = 0.25f + (progress / 100.0f * 0.75f); // Scale between 0.25 and 1.0
            }
            
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        
        // Performance mode checkbox
        checkBoxPerformanceMode.setOnCheckedChangeListener((buttonView, isChecked) -> {
            performanceMode = isChecked;
        });
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
                
                // Select back camera
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
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
    
    private void captureAndAnalyze() {
        if (imageCapture == null) {
            Toast.makeText(this, "Camera not initialized", Toast.LENGTH_SHORT).show();
            return;
        }
        
        if (faceRecognitionHelper == null) {
            Toast.makeText(this, "Face recognition not available", Toast.LENGTH_SHORT).show();
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
        
        // Capture the image
        imageCapture.takePicture(outputFileOptions, executor, new ImageCapture.OnImageSavedCallback() {
            @Override
            public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                // Load the captured image
                Bitmap bitmap = BitmapFactory.decodeFile(outputFile.getAbsolutePath());
                
                // Detect faces
                runOnUiThread(() -> {
                    // Show the captured image
                    viewFinder.setVisibility(View.GONE);
                    imageResult.setVisibility(View.VISIBLE);
                    imageResult.setImageBitmap(bitmap);
                    
                    // Disable the capture button while processing
                    buttonCaptureAnalyze.setEnabled(false);
                    buttonCaptureAnalyze.setText("Processing...");
                });
                
                // Check if face recognition helper is still available
                if (faceRecognitionHelper == null) {
                    runOnUiThread(() -> {
                        buttonCaptureAnalyze.setEnabled(true);
                        buttonCaptureAnalyze.setText("Capture & Analyze");
                        Toast.makeText(MainActivity.this, "Face recognition not available", Toast.LENGTH_SHORT).show();
                    });
                    return;
                }
                
                try {
                    // Detect faces in the image
                    faceRecognitionHelper.detectFaces(bitmap, new FaceRecognitionHelper.FaceDetectionListener() {
                        @Override
                        public void onFaceDetectionSuccess(List<Face> faces, Bitmap originalImage) {
                            // Recognize the detected faces
                            faceRecognitionHelper.recognizeFaces(originalImage, faces, new FaceRecognitionHelper.FaceRecognitionListener() {
                                @Override
                                public void onFaceRecognitionSuccess(List<FaceRecognitionHelper.RecognizedFace> recognizedFaces) {
                                    // Draw face boxes and names on the image
                                    Bitmap resultBitmap = faceRecognitionHelper.drawFaceBoxes(originalImage, recognizedFaces);
                                    
                                    // Prepare results text
                                    StringBuilder resultsText = new StringBuilder();
                                    int unknownCount = 0;
                                    
                                    for (FaceRecognitionHelper.RecognizedFace face : recognizedFaces) {
                                        if (face.getName().equals("Unknown")) {
                                            unknownCount++;
                                        } else {
                                            resultsText.append(face.getName())
                                                    .append(": ")
                                                    .append(String.format("%.2f", face.getConfidence() * 100))
                                                    .append("% confidence\n");
                                        }
                                    }
                                    
                                    if (unknownCount > 0) {
                                        resultsText.append("Unknown faces: ").append(unknownCount).append("\n");
                                    }
                                    
                                    // Update UI
                                    runOnUiThread(() -> {
                                        imageResult.setImageBitmap(resultBitmap);
                                        resultPanel.setVisibility(View.VISIBLE);
                                        textResults.setText(resultsText.toString());
                                        buttonCaptureAnalyze.setEnabled(true);
                                        buttonCaptureAnalyze.setText("Capture & Analyze");
                                    });
                                }
                                
                                @Override
                                public void onFaceRecognitionFailure(String error) {
                                    Log.e(TAG, "Face recognition failed: " + error);
                                    runOnUiThread(() -> {
                                        Toast.makeText(MainActivity.this, "Face recognition failed: " + error, Toast.LENGTH_SHORT).show();
                                        buttonCaptureAnalyze.setEnabled(true);
                                        buttonCaptureAnalyze.setText("Capture & Analyze");
                                    });
                                }
                            });
                        }
                        
                        @Override
                        public void onFaceDetectionFailure(String error) {
                            Log.e(TAG, "Face detection failed: " + error);
                            runOnUiThread(() -> {
                                Toast.makeText(MainActivity.this, "Face detection failed: " + error, Toast.LENGTH_SHORT).show();
                                buttonCaptureAnalyze.setEnabled(true);
                                buttonCaptureAnalyze.setText("Capture & Analyze");
                            });
                        }
                    });
                } catch (Exception e) {
                    Log.e(TAG, "Error in face detection/recognition: " + e.getMessage());
                    runOnUiThread(() -> {
                        Toast.makeText(MainActivity.this, "Error: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                        buttonCaptureAnalyze.setEnabled(true);
                        buttonCaptureAnalyze.setText("Capture & Analyze");
                    });
                }
            }
            
            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                Log.e(TAG, "Image capture failed: " + exception.getMessage());
                runOnUiThread(() -> {
                    Toast.makeText(MainActivity.this, "Image capture failed: " + exception.getMessage(), Toast.LENGTH_SHORT).show();
                    buttonCaptureAnalyze.setEnabled(true);
                    buttonCaptureAnalyze.setText("Capture & Analyze");
                });
            }
        });
    }
    
    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                Toast.makeText(this, R.string.camera_permission_required, Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }
    
    @Override
    protected void onResume() {
        super.onResume();
        
        // Reset UI
        viewFinder.setVisibility(View.VISIBLE);
        imageResult.setVisibility(View.GONE);
        resultPanel.setVisibility(View.GONE);
        
        // Update settings from UI
        recognitionThreshold = seekBarThreshold.getProgress() / 100.0f;
        detectionScale = 0.25f + (seekBarScale.getProgress() / 100.0f * 0.75f);
        performanceMode = checkBoxPerformanceMode.isChecked();
        
        // Apply settings to face recognition helper
        faceRecognitionHelper.setRecognitionThreshold(recognitionThreshold);
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        
        // Clean up resources
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }
        
        if (faceRecognitionHelper != null) {
            faceRecognitionHelper.close();
        }
    }
} 