import tensorflow as tf
from skimage.transform import resize

def normalize_to_range(matrix, new_min=-1, new_max=1):
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

def unnormalize_data(data):
    data += 1
    data /= 2
    return data

def rotate_images_and_labels(images):
    # Implementation of rotating image and creating associated labels for self-supervised learning
    angles = [0, 90, 180, 270]
    rotated_images = []
    labels = []

    for angle in angles:
        # Rotate images
        rotated = tf.image.rot90(images, k=angle // 90)
        rotated_images.append(rotated)
        # Create labels for rotations
        labels += [angle // 90] * tf.shape(images)[0]
    
    # Concatenate all rotated images and labels
    rotated_images = tf.concat(rotated_images, axis=0)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    
    return rotated_images, labels