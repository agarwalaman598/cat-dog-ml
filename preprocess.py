import cv2              # OpenCV for image processing
import os               # To read files and folders
import numpy as np      # For numerical arrays

# Path to dataset
DATASET_PATH = "dataset"

# Image size (same for all images)
IMG_SIZE = 64

# Lists to store data
X = []   # features (images)
y = []   # labels (0 = cat, 1 = dog)

# Function to process images in a folder
def process_images(folder, label):
    folder_path = os.path.join(DATASET_PATH, folder)

    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)

        # Read image
        img = cv2.imread(img_path)

        # If image is corrupted, skip it
        if img is None:
            continue

        # Resize image to 64x64
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Convert image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Flatten image (64x64 → 4096)
        img = img.flatten()

        # Normalize pixel values (0–255 → 0–1)
        img = img / 255.0

        # Append to dataset
        X.append(img)
        y.append(label)

# Process cats (label = 0)
process_images("cats", 0)

# Process dogs (label = 1)
process_images("dogs", 1)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Print shape to verify
print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)

# Save processed data for reuse
np.save("X.npy", X)
np.save("y.npy", y)
