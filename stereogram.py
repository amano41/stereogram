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

    # Normalize the depth map to the range [0, 1]
    def normalize(array):
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val)

    depthmap_array = normalize(depthmap_array)

    # Convert to RGB if grayscale
    if texture_array.ndim == 2:
        texture_array = np.stack((texture_array,) * 3, axis=-1)

    # Create an empty array for the stereogram
    width = depthmap_array.shape[1] + texture_array.shape[1]
    height = depthmap_array.shape[0]
    channels = texture_array.shape[2]
    stereogram_array = np.zeros_like(
        texture_array,
        shape=(height, width, channels),
        dtype=texture_array.dtype,
    )

    # Generate the stereogram
    for y in range(height):
        for x in range(width):
            if x < texture_array.shape[1]:
                sy = y % texture_array.shape[0]
                sx = x % texture_array.shape[1]
                stereogram_array[y, x] = texture_array[sy, sx]
            else:
                sy = y
                sx = x - texture_array.shape[1]
                offset = int(depthmap_array[sy, sx] * offset_factor)
                stereogram_array[y, x] = stereogram_array[sy, sx + offset]

    # Draw guide markers
    distance = texture_array.shape[1]
    marker_radius = 6
    marker_margin = 4
    marker_band_height = (marker_radius + marker_margin) * 2

    my = marker_band_height // 2
    mx1 = width // 2 - distance // 2
    mx2 = width // 2 + distance // 2

    for y in range(marker_band_height):
        for x in range(width):
            for mx in (mx1, mx2):
                if (x - mx) ** 2 + (y - my) ** 2 < marker_radius**2:
                    stereogram_array[y, x] = [0, 0, 0]
                    break

    # Convert back to an image
    stereogram_image = Image.fromarray(stereogram_array)

    # Save the stereogram
    stereogram_image.save(output_path)


if __name__ == "__main__":

    depthmap_path = sys.argv[1]
    texture_path = sys.argv[2]
    output_path = sys.argv[3]

    generate_stereogram(depthmap_path, texture_path, output_path)
