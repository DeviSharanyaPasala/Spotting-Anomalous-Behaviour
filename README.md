# Spotting Anomalous Behaviour

Anomaly detection in surveillance video using a ConvLSTM Autoencoder. The model learns what normal pedestrian movement looks like, then flags frames where the reconstruction error is unusually high.

## How it works

- Input is a sequence of 10 consecutive grayscale frames (227x227)
- A ConvLSTM Autoencoder learns to reconstruct normal motion patterns
- At inference time, frames with reconstruction error above mean + 2 standard deviations are flagged as anomalous
- Tested on the UCSD Anomaly Detection Dataset (Ped1 subset)

## Results

- Final MSE loss: 0.004 to 0.006 after 5 epochs
- Successfully detected anomalous frames in test clips (e.g. frames 34, 35, 36, 70, 71, 72)
- Reconstruction error plots clearly separate normal from abnormal activity

## Dataset

UCSD Anomaly Detection Dataset (Ped1 subset)
- Training: clips with normal pedestrian activity (Train001 to Train010)
- Testing: clips with both normal and abnormal activity (Test001 to Test012)
- Frames are .tif image sequences resized to 227x227

## Stack

Python, TensorFlow, Keras, OpenCV, NumPy, Matplotlib
