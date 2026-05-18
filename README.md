# Spotting Anomalous Behaviour

Detecting unusual activity in surveillance footage using a ConvLSTM Autoencoder trained only on normal pedestrian movement.

The idea is straightforward: if the model only ever sees normal activity during training, it gets good at reconstructing normal frames. When it sees something unusual at test time, the reconstruction error spikes. That spike is how anomalies get flagged.

## How it works

- Input is a sequence of 10 consecutive grayscale frames at 227x227
- A ConvLSTM Autoencoder learns to reconstruct normal motion patterns
- At inference, frames where reconstruction error exceeds the threshold (mean + 2 standard deviations) are flagged as anomalous
- No labels needed during training — fully unsupervised

## Training

- Dataset: UCSD Anomaly Detection Dataset, Ped1 subset
- Training clips: Train001, Train002, Train003 (165 total frames, 155 sequences of 10)
- Epochs: 5
- Loss per epoch: 0.0330, 0.0039, 0.0015, 0.0011, 0.0007
- Final MSE loss: 0.0007

## Results

- Anomalous frames detected: indices 133 to 141
- The reconstruction error plot clearly separates normal from anomalous frames
- Model converged quickly — loss dropped by 97% from epoch 1 to epoch 5

## Dataset

UCSD Anomaly Detection Dataset, Ped1 subset. Training uses clips of normal pedestrian activity. Testing uses clips with a mix of normal and abnormal behavior.

Source: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm

## How to run

```bash
pip install -r requirements.txt
jupyter notebook Spotting_Anomalous_Behaviour.ipynb
```

## Stack

Python, TensorFlow, Keras, OpenCV, NumPy, Matplotlib
