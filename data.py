import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import random
import re
import cv2
from argparse import ArgumentParser
import sys

def load_data(img_size = (256,256), base_dir = './img/dataset/', clean_dir = 'clean', noisy_dirs = ['noisy_1', 'noisy_3', 'noisy_5', 'noisy_7', 'noisy_9'], data_augmentation=False):
    
    # To avoid leaking, the report images won't be used for training
    report_images = [
        'noisy_1_50.png',
        'noisy_3_50.png',
        'noisy_5_50.png',
        'noisy_7_50.png',
        'noisy_9_50.png',
        'noisy_1_100.png',
        'noisy_3_100.png',
        'noisy_5_100.png',
        'noisy_7_100.png',
        'noisy_9_100.png']

    # Image pattern
    img_pattern = r'_(\d+)\.png'

    # Initialize lists to store the images
    X_train = []  # Noisy train images
    Y_train = []  # Clean train images
    X_test = [] # Noisy test images
    Y_test = [] # Clean test images

    # Function to load images
    def load_noisy_and_clean_images(noisy_dir, clean_dir):
        train_noisy_list = []
        train_clean_list = []
        test_noisy_list = []
        test_clean_list = []
        for filename in os.listdir(noisy_dir):
            # Check if the image is test
            is_test = filename in report_images

            if filename.endswith(".png"):
                matches = re.search(img_pattern, filename)
                clean_img = io.imread(os.path.join(clean_dir, 'clean_'+matches.group(1)+'.png'), as_gray=True)
                clean_img = cv2.resize(clean_img, img_size)
                noisy_img = io.imread(os.path.join(noisy_dir, filename), as_gray=True)
                noisy_img = cv2.resize(noisy_img, img_size)
                if is_test:
                    test_clean_list.append(clean_img)
                    test_noisy_list.append(noisy_img)
                else:
                    train_clean_list.append(clean_img)
                    train_noisy_list.append(noisy_img)
        return (train_noisy_list,train_clean_list,test_noisy_list,test_clean_list)

    # Load noisy images and corresponding clean images from each directory
    for noisy_dir in noisy_dirs:
        (train_noisy,train_clean,test_noisy,test_clean) = load_noisy_and_clean_images(base_dir + noisy_dir, base_dir + clean_dir)
        X_train.extend(train_noisy)
        Y_train.extend(train_clean)
        X_test.append(test_noisy)
        Y_test.append(test_clean)

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
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    # reshape test arrays
    X_test = X_test.reshape(X_test.shape[0] * X_test.shape[1], X_test.shape[2], X_test.shape[3])
    Y_test = Y_test.reshape(Y_test.shape[0] * Y_test.shape[1], Y_test.shape[2], Y_test.shape[3])

    # Convert to float and normalize
    X_train = X_train/255.0
    Y_train = Y_train/255.0
    X_test = X_test/255.0
    Y_test = Y_test/255.0

    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--print-test', type=bool, default=False, help='print train images randomly')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    X_train, Y_train, X_test, Y_test = load_data()
    print("X_train and Y_train shapes:")
    print(X_train.shape)
    print(Y_train.shape)
    print("X_test and Y_test shapes:")
    print(X_test.shape)
    print(Y_test.shape)

    while not args.print_test:
        random_index = random.randint(0, len(X_train) - 1)
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Noisy Image")
        plt.imshow(X_train[random_index], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Clean Image")
        plt.imshow(Y_train[random_index], cmap='gray')
        plt.show()
    
    for i in range(len(X_test)):
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        img_number = str((i%(len(X_test)//2))*2 + 1)
        plt.title("Noisy Image " + img_number)
        plt.imshow(X_test[i], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Clean Image " + img_number)
        plt.imshow(Y_test[i], cmap='gray')
        plt.show()