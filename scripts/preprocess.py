import numpy as np
import os
import random
from sklearn.model_selection import train_test_split

raw = './data/raw'
output = './data/processed'
samples = 1500 # number of images per class

files = [f for f in os.listdir(raw) if f.endswith('.npy')]
files.sort()

X_list = []
y_list = []
class_names = []

for index, file_name in enumerate(files):
    class_name = file_name.replace('.npy', '') # Remove file extension to get class
    class_names.append(class_name)

    file_path = os.path.join(raw, file_name)
    class_data = np.load(file_path) # Load array of class images

    num_images = class_data.shape[0]  # Total number images in class

    if num_images > samples:
        sampled_indices = random.sample(range(num_images), samples) # Get random images
        class_data = class_data[sampled_indices]
    
    X_list.append(class_data)
    labels = np.full(class_data.shape[0], index) # Create labels for class
    y_list.append(labels)

X_combined = np.vstack(X_list) # Stack all image arrays vertically
y_combined = np.concatenate(y_list) # Concatenate all label arrays

shuffle_indices = np.random.permutation(len(X_combined)) # Create a random arrangement of indices
X_shuffled = X_combined[shuffle_indices] # Shuffle datast
y_shuffled = y_combined[shuffle_indices]

X = X_shuffled.reshape(-1, 28, 28, 1) # Reshape images for cnn
y = y_shuffled

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Train split
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp) # Validation and test split

np.savez_compressed(os.path.join(output, 'train.npz'), X=X_train, y=y_train, class_names=class_names)
np.savez_compressed(os.path.join(output, 'val.npz'), X=X_val, y=y_val, class_names=class_names)
np.savez_compressed(os.path.join(output, 'test.npz'), X=X_test, y=y_test, class_names=class_names)