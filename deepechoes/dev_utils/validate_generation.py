from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
from PIL import Image
import numpy as np

def load_spectrogram(image_path, image_shape):
    """Load spectrogram image and convert to grayscale array."""
    img = Image.open(image_path).resize(image_shape)
    return np.array(img)

def filter_spectrograms(spectrograms, real_pca_scores, pca):
    """
    Filter spectrograms using PCA-based criteria (IQR + mean/std).
    
    Parameters:
    - spectrograms: Array of generated spectrograms.
    - real_pca_scores: PCA scores of real spectrograms (for reference).
    - pca: PCA model trained on real spectrograms.
        
    Returns:
    - filtered_spectrograms: List of spectrograms that pass the filter.
    """
    
    gen_pca_scores = pca.transform(spectrograms.reshape(len(spectrograms), -1))

    # 2D Filtering Using Mahalanobis Distance
    mean_real_pca = np.mean(real_pca_scores, axis=0)
    cov_real_pca = np.cov(real_pca_scores, rowvar=False)
    cov_inv_real_pca = inv(cov_real_pca)

    distances = [
        mahalanobis(score, mean_real_pca, cov_inv_real_pca)
        for score in gen_pca_scores
    ]

    # Define a threshold for filtering (e.g., 95% of real spectrograms)
    threshold = np.percentile(
        [mahalanobis(score, mean_real_pca, cov_inv_real_pca) for score in real_pca_scores], 95
    )

    # Filter generated spectrograms based on Mahalanobis distance
    keep_indices = np.array(distances) <= threshold
    
    return spectrograms[keep_indices], keep_indices