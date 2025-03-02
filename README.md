# ECG Arrhythmia Detection using Deep Learning

## Project Overview

This project implements advanced deep learning techniques (LSTM, CNN and AlexNet) for detecting heart arrhythmias using the St. Petersburg INCART 12-lead Arrhythmia Database. The goal is to develop robust machine learning models that can accurately identify abnormal heart rhythms from ECG signals.

## Dataset

- **Source**: St. Petersburg INCART 12-lead Arrhythmia Database
- **Characteristics**:
  - 75 subjects with 30-minute ECG recordings
  - Sampled at 257 Hz
  - Annotated by medical experts for arrhythmia detection

## Key Features

### Data Preprocessing

- Custom data extraction function to segment ECG signals
- Feature extraction using Short-Time Fourier Transform (STFT)
- Handling class imbalance with custom loss function

### Model Architectures

The project implements three different neural network architectures:

1. **Bidirectional LSTM**

   - Captures temporal dependencies in ECG signals
   - Processes sequential data with bidirectional context

2. **1D Convolutional Neural Network (CNN)**

   - Extracts spatial patterns from ECG signals
   - Uses convolutional and pooling layers for feature detection

3. **AlexNet-Inspired Architecture**
   - Adapted from the original AlexNet image classification model
   - Customized for 1D ECG signal processing

### Key Techniques

- Custom Binary Cross-Entropy (BCE) loss function
- Handling class imbalance
- Data visualization of ECG signals
- Performance evaluation using accuracy and F1 score

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-learn
- WFDB (Waveform Database Software Package)

## Results

The project compares the performance of different neural network architectures:

- Bidirectional LSTM
- 1D Convolutional Neural Network
- AlexNet-inspired Architecture

Each model is evaluated using:

- Training accuracy
- Test accuracy
- F1 score

## Key Insights

- The dataset shows a significant class imbalance (only 13.3% abnormal samples)
- Custom loss function helps address class imbalance
- CNN outperformed LSTM in this specific classification task

## Future Work

- Experiment with more advanced architectures
- Incorporate additional feature extraction techniques
- Collect more diverse ECG data

## Acknowledgments

- St. Petersburg INCART for providing the ECG database
- PyTorch and scientific computing community
