import numpy as np

def scale_to_range(matrix, new_min=-1, new_max=1):
    """
    Normalize the input matrix to a specified range [new_min, new_max].

    Parameters:
    - matrix: Input data.
    - new_min, new_max: The target range for normalization.

    Returns:
    - Normalized data scaled to the range [new_min, new_max].
    """
    original_max = matrix.max()
    original_min = matrix.min()
    # Scale the matrix to [0, 1]
    normalized = (matrix - original_min) / (original_max - original_min)
    # Scale and shift to [new_min, new_max]
    scaled = normalized * (new_max - new_min) + new_min
    
    return scaled

def tonal_noise_reduction(data):
    row_medians = np.median(data, axis=1, keepdims=True)
    col_medians = np.median(data, axis=0, keepdims=True)
    spec_normalized = data - row_medians - col_medians
    return spec_normalized

def normalize_to_zero_mean_unit_variance(data, clip_std=False):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    normalized_data = (data - mean) / std
    if clip_std:
        normalized_data /= 3
    return normalized_data

def unscale_data(data, min_val=0, max_val=1):
    data += 1
    data /= 2
    return data * (max_val - min_val) + min_val

def rotate_images_and_labels(images):
    # Implementation of rotating image and creating associated labels for self-supervised learning
    angles = [0, 90, 180, 270]
    rotated_images = []
    labels = []

    for angle in angles:
        # Rotate images
        rotated = np.rot90(images, k=angle // 90, axes=(1,2))
        rotated_images.append(rotated)
        # Create labels for rotations
        labels += [angle // 90] * images.shape[0]
    
    # Concatenate all rotated images and labels
    rotated_images = np.concatenate(rotated_images, axis=0)
    labels = np.array(labels, dtype=np.int32)
    
    return rotated_images, labels