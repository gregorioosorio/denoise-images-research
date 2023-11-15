import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import random
import re
import cv2

def load_data(img_size = (256,256), base_dir = './img/dataset/', clean_dir = 'clean', noisy_dirs = ['noisy_1', 'noisy_3', 'noisy_5', 'noisy_7', 'noisy_9'], data_augmentation=False):
    
    # To avoid leaking, we will ignore the images that are used in the report
    ignore_images = [
        'noisy_1_50.png', 'noisy_1_100.png',
        'noisy_3_50.png', 'noisy_3_100.png',
        'noisy_5_50.png', 'noisy_5_100.png',
        'noisy_7_50.png', 'noisy_7_100.png',
        'noisy_9_50.png', 'noisy_9_100.png']

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
            # If filename is in the report images list, then ignore it
            if filename in ignore_images:
                print('ignoring: ', filename)
                continue

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

    def flip(image):
        image = cv2.flip(image, 1)  # 1 for horizontal flip
        return image

    def rotate(image, angle):
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        return image

    def augment(X, Y):
        X_augmented = []
        Y_augmented = []
        for i in range(len(X)):
            X_augmented.append(X[i])
            Y_augmented.append(Y[i])
            if np.random.rand() > 0.5:
                X_augmented.append(flip(X[i]))
                Y_augmented.append(flip(Y[i]))
            else:
                angle = np.random.randint(-45, 46)  # Rotate by a random angle between -45 and 45 degrees
                X_augmented.append(rotate(X[i], angle))
                Y_augmented.append(rotate(Y[i], angle))
        return X_augmented, Y_augmented

    if data_augmentation:
         # Augment the data
         X_train, Y_train = augment(X_train, Y_train)

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
    while True:
        random_index = random.randint(0, len(X_train) - 1)
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Noisy Image")
        plt.imshow(X_train[random_index], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Clean Image")
        plt.imshow(Y_train[random_index], cmap='gray')
        plt.show()