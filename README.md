# Spotting Anomalous Behaviour Using Camera-Based System

**Author:** Devi Sharanya Pasala

**Department of Information Science and Technology, University at Albany, SUNY, NY, USA**

**Email:** [dpasala@albany.edu](mailto:dpasala@albany.edu)



## Project Overview

This project focuses on developing an automated system for detecting anomalous human behaviors in surveillance videos using deep learning.  
The model leverages a **Convolutional Long Short-Term Memory (ConvLSTM) Autoencoder** to learn normal motion patterns and identify unusual activities that deviate from the expected behavior.

Traditional surveillance systems rely heavily on human observation, which is time-consuming and prone to error. This project aims to automate this process using deep learning–based spatio-temporal modeling, providing a scalable and intelligent approach for anomaly detection.



## Objectives

* To design a **ConvLSTM Autoencoder** capable of learning spatio-temporal features from video frames.
* To detect anomalous movements by identifying high reconstruction errors.
* To enhance public safety monitoring through intelligent automation.
* To visualize anomalies using reconstruction error thresholds.



## Models and Methods

The anomaly detection system is based on a **ConvLSTM Autoencoder**, which captures both spatial and temporal dependencies in video data.

**Model Architecture:**
* **Input:** 10 consecutive grayscale frames (227×227 pixels each)
* **Encoder:** Two ConvLSTM layers (64 and 32 filters) for temporal feature extraction
* **Decoder:** 3D convolution layer with sigmoid activation for sequence reconstruction
* **Loss Function:** Mean Squared Error (MSE)
* **Optimizer:** Adam optimizer with a learning rate of 0.001

The model was implemented using **TensorFlow** and **Keras**, trained on normal behavior sequences, and evaluated on test sequences containing anomalies.



## Dataset

* **Source:** UCSD Anomaly Detection Dataset (UCSD Ped1 subset)
* **Location:** Google Drive → `undergrad major project/Spotting Anomalous Behaviour.zip`
* **Content:**  
  * **Training Data:** Normal clips (`Train001 – Train010`)
  * **Testing Data:** Clips with both normal and abnormal behavior (`Test001 – Test012`)
* **Format:** `.tif` image sequences (individual frames)
* **Resolution:** 158×238 pixels (resized to 227×227 before training)



## Implementation Steps

1. **Data Extraction and Preprocessing**

   * Dataset extracted from Google Drive.
   * Frames resized to 227×227 and converted to grayscale.
   * Normalized pixel intensity between 0 and 1.
   * Created frame sequences of 10 frames each.

2. **Model Training**

   * Trained on normal activity sequences using ConvLSTM Autoencoder.
   * Used MSE loss and Adam optimizer.
   * Trained for 5 epochs with batch size of 2.

3. **Testing and Evaluation**

   * Model reconstructs sequences from test videos.
   * High reconstruction error indicates abnormal activity.
   * Threshold = Mean + 2×Standard Deviation of reconstruction error.

4. **Visualization**

   * Reconstruction error plotted for each frame index.
   * Frames exceeding the threshold are marked as anomalies.



## Results

| Metric       | Description                                   | Example Value      |
|--------------|-----------------------------------------------|--------------------|
| **Loss**     | Mean Squared Error after final epoch          | 0.004 – 0.006      |
| **Threshold**| Adaptive threshold (Mean + 2×Std deviation)   | Computed per test  |
| **Output**   | Frames detected as anomalies                  | [34, 35, 36, 70, 71, 72] |

The results clearly demonstrate that frames containing unusual movements exhibit higher reconstruction errors, allowing effective anomaly detection.



## Key Findings

* ConvLSTM Autoencoder accurately models normal pedestrian movement patterns.
* Anomalous frames are effectively detected by comparing reconstruction errors.
* Integrating the dataset through Google Drive ensures stable execution and reproducibility.
* Increasing the number of epochs and using additional training clips can improve performance.



## Conclusion

The study concludes that the **ConvLSTM Autoencoder** provides a robust and efficient solution for anomaly detection in video surveillance.  
By learning temporal and spatial features simultaneously, the model can successfully identify abnormal behaviors in crowded environments.  
This framework can be extended to real-time surveillance systems, improving security monitoring through automation and reducing dependence on manual observation.



## Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Matplotlib
* Google Colab
* Google Drive Integration



## Contact

For questions or collaboration, reach out:  
**Devi Sharanya Pasala**  
[dpasala@albany.edu](mailto:dpasala@albany.edu)



## References

* UCSD Anomaly Detection Dataset — University of California San Diego  
* Shi et al. (2015). *A Convolutional LSTM Network for Spatiotemporal Sequence Forecasting.*  
* Related works from *IEEE*, *Elsevier*, and *Springer* journals referenced in the project report.



### If you find this project useful, please consider giving it a star on GitHub!
