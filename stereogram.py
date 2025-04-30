import sys

import numpy as np
from PIL import Image


def generate_stereogram(depthmap_path, texture_path, output_path, offset_factor=20):

    # Load the image and depth map
    depthmap_image = Image.open(depthmap_path).convert("L")  # Convert to grayscale
    texture_image = Image.open(texture_path)

    # Convert images to numpy arrays
    depthmap_array = np.array(depthmap_image)
    texture_array = np.array(texture_image)
    # Ensure the depth map is in grayscale

    # Normalize the depth map to the range [0, 1]
    def normalize(array):
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val)

    depthmap_array = normalize(depthmap_array)

    if texture_array.ndim == 2:
        texture_array = np.stack((texture_array,) * 3, axis=-1)  # Convert to RGB if grayscale

    # Create an empty array for the stereogram
    stereogram_array = np.zeros_like(
        texture_array,
        shape=(depthmap_array.shape[0], depthmap_array.shape[1] + texture_array.shape[1], texture_array.shape[2]),
        dtype=texture_array.dtype,
    )

    # Generate the stereogram
    for y in range(stereogram_array.shape[0]):
        for x in range(stereogram_array.shape[1]):
            if x < texture_array.shape[1]:
                sy = y % texture_array.shape[0]
                sx = x % texture_array.shape[1]
                stereogram_array[y, x] = texture_array[sy, sx]
            else:
                offset = int(depthmap_array[y, x - texture_array.shape[1]] * offset_factor)
                sx = x - texture_array.shape[1] + offset
                sy = y
                stereogram_array[y, x] = stereogram_array[sy, sx]

    # Convert back to an image
    stereogram_image = Image.fromarray(stereogram_array)

    # Save the stereogram
    stereogram_image.save(output_path)


if __name__ == "__main__":

    depthmap_path = sys.argv[1]
    texture_path = sys.argv[2]
    output_path = sys.argv[3]

    generate_stereogram(depthmap_path, texture_path, output_path)
