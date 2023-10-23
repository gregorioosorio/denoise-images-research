import tensorflow as tf
from argparse import ArgumentParser
import sys
import matplotlib.pyplot as plt
import random
import numpy as np

# local modules
import data
import model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-weights-path', type=str, default='./trained_models/denoise_unet.h5', help='model weights path')
    parser.add_argument('--data-percentage', type=float, default=0.1, help='percentage of the data to predict')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    print('----------------- Denoise Images Using U-Net -----------------')
    print('Author: Gregorio Osorio')
    print('GitHub: gregorioosorio')
    print('--------------------------------------------------------------')

    print('----------------- Loading Data -----------------')
    X_train, Y_train = data.load_data()

    print('----------------- Loading Weights -----------------')
    model = model.u_net((256,256,1))
    model.load_weights(args.model_weights_path)

    print('----------------- Predicting -----------------')
    preds_train = model.predict(X_train[:int(X_train.shape[0]*args.data_percentage)], verbose=1)

    while True:
        fig, axes = plt.subplots(3, 3)

        for i in range(3):
            ix = random.randint(0, len(preds_train) - 1)

            # Plot the images in the subplots
            axes[i][0].imshow(X_train[ix], cmap='gray')
            axes[i][0].set_title('noisy')

            axes[i][1].imshow(preds_train[ix], cmap='gray')
            axes[i][1].set_title('predict')

            axes[i][2].imshow(Y_train[ix], cmap='gray')
            axes[i][2].set_title('clean')

        # Adjust spacing between subplots
        plt.tight_layout()

        # Display the plot
        plt.show()
