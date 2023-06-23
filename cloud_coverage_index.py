import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Directory path containing NIR images
directory = '/Users/jacopobiancat/OneDrive - Istituto Europeo di Studi Superiori/Astro-PI-22_23/EO_IESS'

# Initialize lists to store cloud indices and image names
cloud_indices = []
image_names = []

# Loop through each file in the directory
for file in sorted(os.listdir(directory)):
    if file.endswith('.jpg'):
        # Read the NIR image
        image_path = os.path.join(directory, file)
        nir_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply adaptive thresholding to segment potential cloud pixels
        _, thresholded_image = cv2.threshold(nir_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert the thresholded image to binary format
        cloud_mask = np.where(thresholded_image > 0, 255, 0).astype(np.uint8)

        # Perform morphological operations to refine the cloud mask
        kernel = np.ones((3, 3), np.uint8)
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN, kernel)

        # Calculate the cloud cover percentage
        total_pixels = cloud_mask.size
        cloud_pixels = np.count_nonzero(cloud_mask)
        cloud_cover_percentage = (cloud_pixels / total_pixels) * 100

        # Create an index indicating the quantity of clouds
        cloud_index = cloud_cover_percentage / 100

        # Append the cloud index and image name to the respective lists
        cloud_indices.append(cloud_index)
        image_names.append(file[:-4])  # Remove the file extension

# Plot the cloud indices
plt.figure(figsize=(10, 6))
plt.bar(image_names, cloud_indices)
plt.xticks(rotation=45)
plt.xlabel('Image')
plt.ylabel('Cloud Index')
plt.title('Cloud Index for NIR Images')
plt.tight_layout()
plt.show()

# Plot the distribution of cloud coverage
bins = np.arange(0, 1.1, 0.1)
plt.hist(cloud_indices, bins=bins, edgecolor='black', alpha=0.7)
plt.xticks(bins)
plt.xlabel('Cloud Coverage')
plt.ylabel('Number of Images')
plt.title('Distribution of Cloud Coverage')
plt.tight_layout()
plt.show()