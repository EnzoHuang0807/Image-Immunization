import numpy as np
from scipy.ndimage import label, binary_fill_holes

# Create a binary mask (0 = black, 1 = white)
mask = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0]
])

# Use `binary_fill_holes` to fill enclosed black regions
filled_mask = binary_fill_holes(mask).astype(int)

print("Original mask:")
print(mask)
print("\nFilled mask:")
print(filled_mask)
