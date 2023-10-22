import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import random
import re
import cv2

def load_data(img_size = (256,256), base_dir = './img/dataset/', clean_dir = 'clean', noisy_dirs = ['noisy_1', 'noisy_3', 'noisy_5', 'noisy_7', 'noisy_9']):
    # Image pattern
    img_pattern = r'_(\d+)\.png'

    # Initialize lists to store the images
    X_train = []  # Noisy images
    Y_train = []  # Clean images

    # Function to load images
    def load_noisy_and_clean_images(noisy_dir, clean_dir):
        images_list = []
        clean_list = []
        for filename in os.listdir(noisy_dir):
            if filename.endswith(".png"):
                matches = re.search(img_pattern, filename)
                clean_img = io.imread(os.path.join(clean_dir, 'clean_'+matches.group(1)+'.png'), as_gray=True)
                clean_img = cv2.resize(clean_img, img_size)
                clean_list.append(clean_img)
                noisy_img = io.imread(os.path.join(noisy_dir, filename), as_gray=True)
                noisy_img = cv2.resize(noisy_img, img_size)
                images_list.append(noisy_img)
        return (images_list,clean_list)

    # Load noisy images and corresponding clean images from each directory
    for noisy_dir in noisy_dirs:
        (noisy,clean) = load_noisy_and_clean_images(base_dir + noisy_dir, base_dir + clean_dir)
        X_train.extend(noisy)
        Y_train.extend(clean)

    # Convert the lists of images into numpy arrays
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # Convert to float and normalize
    X_train = X_train/255.0
    Y_train = Y_train/255.0

    return X_train, Y_train

if __name__ == "__main__":
    X_train, Y_train = load_data()
    print(X_train.shape)
    print(Y_train.shape)
    random_index = random.randint(0, len(X_train) - 1)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Noisy Image")
    plt.imshow(X_train[random_index], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Clean Image")
    plt.imshow(Y_train[random_index], cmap='gray')
    plt.show()