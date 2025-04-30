import sys

import numpy as np
from PIL import Image


def generate_whitenoise(width, height, output_path, grayscale=True):

    if grayscale:
        shape = (height, width)
        mode = "L"  # Grayscale mode
    else:
        shape = (height, width, 3)
        mode = "RGB"

    noise_array = np.random.randint(0, 256, shape, dtype=np.uint8)
    noise_image = Image.fromarray(noise_array, mode)
    noise_image.save(output_path)


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: python whitenoise.py <width> <height> <output_path>")
        sys.exit(1)

    width = int(sys.argv[1])
    height = int(sys.argv[2])
    output_path = sys.argv[3]

    generate_whitenoise(width, height, output_path)
