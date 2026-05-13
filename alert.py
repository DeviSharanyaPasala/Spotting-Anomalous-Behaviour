import numpy as np

def check_anomaly(reconstruction_errors, threshold=None):
    """
    Checks reconstruction errors against a threshold.
    Fires an alert if anomalous frames are detected.
    
    Args:
        reconstruction_errors: list of float MSE values per frame
        threshold: optional float, defaults to mean + 2*std
    
    Returns:
        list of anomalous frame indices
    """
    if threshold is None:
        threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)

    anomalies = [i for i, e in enumerate(reconstruction_errors) if e > threshold]

    if anomalies:
        print(f"[ALERT] Anomaly detected at frames: {anomalies}")
        print(f"[INFO]  Threshold used: {threshold:.6f}")
    else:
        print("[OK] No anomalies detected — all frames within normal range")

    return anomalies


if __name__ == "__main__":
    sample_errors = np.random.normal(loc=0.004, scale=0.001, size=100)
    sample_errors[34] = 0.02   # inject anomaly
    sample_errors[70] = 0.025  # inject anomaly
    check_anomaly(sample_errors.tolist())
