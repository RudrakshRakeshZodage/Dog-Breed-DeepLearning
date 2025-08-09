# Dog_Breed_DeepLearning
---------------------------------------
#Overview
- This project is a deep learning-based image classification system designed to identify different dog breeds from images.
- It leverages TensorFlow and MobileNetV2 for transfer learning, allowing efficient training even on limited hardware.
- The model supports GPU acceleration if available but automatically falls back to CPU for compatibility.

---------------------------------------
# Features
- Automatic GPU/CPU Selection — Uses GPU if available, else runs on CPU.
- Transfer Learning with MobileNetV2 — Faster training with pre-trained ImageNet weights.
- Data Augmentation — Horizontal flips and zoom to improve generalization.
- Multi-class Classification — Supports multiple dog breeds with categorical output.
- Model Saving — Stores trained model (.h5) and class labels (.npy).
---------------------------------------
# System Workflow
- Step 1: Load training and validation datasets from pre-split folders.
- Step 2: Apply image preprocessing and augmentation using ImageDataGenerator.
- Step 3: Initialize MobileNetV2 as the feature extractor and freeze its weights.
- Step 4: Add fully connected layers and dropout for classification.
- Step 5: Compile and train the model using categorical crossentropy loss.
- Step 6: Save the trained model and class label mapping for future predictions.
 ---------------------------------------
# Technology Stack
- Python — Core programming language.
- TensorFlow / Keras — Deep learning framework.
- MobileNetV2 — Pre-trained CNN backbone for feature extraction.
- NumPy — For handling and saving label arrays.
- ImageDataGenerator — Data preprocessing and augmentation.
---------------------------------------
# Performance
- Training can utilize GPU acceleration for faster computation.
- Uses transfer learning to significantly reduce training time and improve accuracy.
- Suitable for small-to-medium datasets due to its efficient architecture.
