import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim

def contrast_stretch(im):
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out

def calc_ndvi(image):
    b, g, r = cv2.split(image)
    bottom = (r.astype(float) + b.astype(float))
    bottom[bottom==0] = 0.01
    ndvi = (b.astype(float) - r) / bottom
    # Normalize NDVI to range [0, 1]
    min_ndvi = np.min(ndvi)
    max_ndvi = np.max(ndvi)
    normalized_ndvi = (ndvi - min_ndvi) / (max_ndvi - min_ndvi)

    return normalized_ndvi

# Directory path containing NIR images
directory = '/Users/jacopobiancat/OneDrive - Istituto Europeo di Studi Superiori/Astro-PI-22_23/EO_IESS'

# Path to the CSV file containing latitude and longitude information
csv_file = '/Users/jacopobiancat/Documents/Scuola/Astro-PI-22-23/data_with_countries.csv'

# Initialize geocoder
geolocator = Nominatim(user_agent='my_app')

# Read the CSV file
data = pd.read_csv(csv_file)

# Initialize lists to store water indices and country names
ndvi_min = []
ndvi_max = []
ndvi_mean = []
ndvi_std_dev = []
countries = data['Country']

# Loop through each file in the directory
for file in sorted(os.listdir(directory)):
    if file.endswith('.jpg'):
        # Read the NIR image
        image_path = os.path.join(directory, file)
        print(image_path)
        nir_image = cv2.imread(image_path)

        # Calculate the NDVI
        contrasted = contrast_stretch(nir_image)
        normalized_ndvi = calc_ndvi(contrasted)
        #ndvi_contrasted = contrast_stretch(normalized_ndvi)

        # Calculate the mean
        mean = np.mean(normalized_ndvi)
        # Calculate the standard deviation
        std_dev = np.std(normalized_ndvi)

        # Append the min and max values of NDVI to the lists
        ndvi_mean.append(mean)
        ndvi_std_dev.append(std_dev)

# Create a dataframe with the NDVI indices and country names
df = pd.DataFrame({'Country': countries, 'NDVI': ndvi_mean})

# Calculate the average water index per country
average_NDVI_mean = df.groupby('Country')['NDVI'].mean()

# Plot the average water index per country
plt.figure(figsize=(10, 6))
average_NDVI_mean.plot(kind='bar', edgecolor='black', alpha=0.7)
plt.xlabel('Country')
plt.ylabel('Average NDVI')
plt.title('Average NDVI per Country')
plt.tight_layout()
plt.show()

# Create a dataframe with the NDVI indices and country names
df = pd.DataFrame({'Country': countries, 'NDVI': ndvi_std_dev})

# Calculate the average water index per country
average_NDVI_std = df.groupby('Country')['NDVI'].mean()

# Plot the average water index per country
plt.figure(figsize=(10, 6))
average_NDVI_std.plot(kind='bar', edgecolor='black', alpha=0.7)
plt.xlabel('Country')
plt.ylabel('Average NDVI Standard Deviation')
plt.title('Average NDVI Standard Deviation per Country')
plt.tight_layout()
plt.show()