# Spotting Anomalous Behaviour

Detecting unusual activity in surveillance footage using a ConvLSTM Autoencoder. The model learns what normal pedestrian movement looks like and flags frames where it can't reconstruct the sequence accurately.

## How it works

Normal video clips are used for training only. The autoencoder learns to reconstruct normal motion patterns. At test time, frames with reconstruction error above a threshold (mean + 2 standard deviations) are flagged as anomalies. No labels are needed during training since it's an unsupervised approach.

## Model

- Architecture: ConvLSTM Autoencoder
- Input: 10 consecutive grayscale frames at 227x227 pixels
- Encoder: two ConvLSTM layers with 64 and 32 filters
- Decoder: 3D convolution with sigmoid activation
- Loss: Mean Squared Error
- Optimizer: Adam (lr 0.001)
- Training: 5 epochs, batch size 2

## Dataset

UCSD Anomaly Detection Dataset, Ped1 subset.
- Training: 10 clips of normal pedestrian activity (Train001 to Train010)
- Testing: 12 clips with normal and abnormal activity mixed (Test001 to Test012)
- Format: .tif image sequences, resized to 227x227

## Results

- Final MSE loss: 0.004 to 0.006
- Anomalous frames correctly detected in test clips (e.g. frames 34, 35, 36, 70, 71, 72)
- Reconstruction error plots clearly separate normal from abnormal frames
- Threshold computed per test sequence based on mean and standard deviation

## Stack

Python, TensorFlow, Keras, OpenCV, NumPy, Matplotlib
