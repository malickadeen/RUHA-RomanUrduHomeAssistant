# RUHA-RomanUrduHomeAssistant
Developed a home automation system that responds to voice commands in Roman Urdu. Trained with phrases like 'Ruha, turn off the room light,' facilitating effortless control of household devices. Demonstrates expertise in language comprehension, enhancing accessibility to technology for non-English speakers
# Features
1.Voice Command Recognition: Utilizes machine learning models to recognize and understand voice commands spoken in Roman Urdu.
2.Multitask Support: Supports a variety of household tasks including controlling lights, air conditioning, music playback, and television operations.
3.Flexible Architecture: Designed with modularity in mind, allowing for easy integration of new devices and functionalities.
4.Customizable: Users can train the system with their own voice commands and preferences.

# Project Structure
The project consists of two main components:

1. Training Component:
Python scripts responsible for training machine learning models using audio samples of voice commands. This component preprocesses the audio data, extracts relevant features such as MFCC (Mel-Frequency Cepstral Coefficients) and Chromagrams, and trains multiple classification models including Decision Trees, k-Nearest Neighbors (KNN), Support Vector Machines (SVM), AdaBoost, Gaussian Naive Bayes (GNB), and Extra Trees Classifier (ETC). The trained models are then serialized and saved for later use.

2. Inference Component: Python script for real-time inference using trained models. Given an audio input, this component extracts features from the input audio, feeds them into the trained models, and predicts the corresponding action or device control. The predicted actions are then executed accordingly.


# Training
To train the models, follow these steps:

Prepare your audio dataset with voice commands labeled accordingly.
Run the training script (train.py) providing the path to the dataset.
Trained models will be saved as serialized files for later use.
Inference
To perform real-time inference, follow these steps:

Provide the path to the audio file containing the voice command you want to execute.
Run the inference script (inference.py).
The system will output the predicted action based on the trained models.

# Dependencies
Python 3.x
Libraries: librosa, soundfile, pandas, numpy, scikit-learn, imbalanced-learn

# Contributors
malickadeen@gmail.com
