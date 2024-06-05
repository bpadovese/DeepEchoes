import numpy as np
import pytest
from deepechoes.utils.image_transforms import rotate_images_and_labels

def test_rotate_images_and_labels():
  batch_size, height, width, channels = 2, 4, 4, 3 
  dummy_images = np.random.rand(batch_size, height, width, channels)

  rotated_images, labels = rotate_images_and_labels(dummy_images)

  # Check the shape of rotated images
  assert rotated_images.shape == (batch_size * 4, height, width, channels), "Rotated images shape mismatch"

  # Check the shape of labels
  assert labels.shape == (batch_size * 4,), "Labels shape mismatch"

  # Check the values of labels
  expected_labels = [0, 0, 1, 1, 2, 2, 3, 3]
  assert np.array_equal(labels, expected_labels), "Labels values mismatch"


def test_rotate_images():
  # shape (1, 3, 3, 1)
  images = np.array([[[[0], [1], [2]],
                        [[3], [4], [5]],
                        [[6], [7], [8]]]])
  
  rotated_images, _ = rotate_images_and_labels(images)

  # (4,3,3,1)
  expected_rotated_images = np.array([
        [[[0], [1], [2]],
         [[3], [4], [5]],
         [[6], [7], [8]]],
        [[[2], [5], [8]],
         [[1], [4], [7]],
         [[0], [3], [6]]],
        [[[8], [7], [6]],
         [[5], [4], [3]],
         [[2], [1], [0]]],
        [[[6], [3], [0]],
         [[7], [4], [1]],
         [[8], [5], [2]]]
    ])


  assert np.array_equal(rotated_images, expected_rotated_images), "Rotated images do not match expected images"