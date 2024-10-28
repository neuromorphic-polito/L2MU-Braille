import numpy as np
import matplotlib.pyplot as plt
import struct

# File path
file_path = 'train-images-idx3-ubyte'

# Read the IDX file
with open(file_path, 'rb') as f:
    # Read magic number, number of images, rows, and columns
    magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
    # Read the image data
    images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)

# Display the first image
plt.imshow(images[0], cmap='gray')
plt.show()