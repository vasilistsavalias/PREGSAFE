import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca_eigen_diff(real_data, synthetic_data):
    """
    Calculates the mean absolute difference between the eigenvalues of the PCA
    of the real and synthetic data.
    """
    scaler = StandardScaler()
    real_data_scaled = scaler.fit_transform(real_data)
    synthetic_data_scaled = scaler.transform(synthetic_data)

    pca_real = PCA()
    pca_real.fit(real_data_scaled)
    eigenvalues_real = pca_real.explained_variance_

    pca_synthetic = PCA()
    pca_synthetic.fit(synthetic_data_scaled)
    eigenvalues_synthetic = pca_synthetic.explained_variance_

    # Pad the shorter array of eigenvalues with zeros
    len_diff = len(eigenvalues_real) - len(eigenvalues_synthetic)
    if len_diff > 0:
        eigenvalues_synthetic = np.pad(eigenvalues_synthetic, (0, len_diff), 'constant')
    elif len_diff < 0:
        eigenvalues_real = np.pad(eigenvalues_real, (0, -len_diff), 'constant')

    return np.mean(np.abs(eigenvalues_real - eigenvalues_synthetic))

def convert_numpy_to_native(obj):
    """
    Recursively converts numpy data types to native Python types in a dictionary or list.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(i) for i in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj