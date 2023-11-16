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
    parser.add_argument('--model-variant', type=str, default='u_net', help='model variant: u_net, u_net_gn, u_net_res')


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
    _, _, X_test, Y_test = data.load_data()

    print('----------------- Loading Weights -----------------')
    if args.model_variant == 'u_net':
        model = model.u_net((256,256,1))
    elif args.model_variant == 'u_net_gn':
        model = model.u_net_gn((256,256,1))
    elif args.model_variant == 'u_net_res':
        model = model.u_net_res((256,256,1))

    model.load_weights(args.model_weights_path)

    print('----------------- Predicting -----------------')
    preds_train = model.predict(X_test[:int(X_test.shape[0]*args.data_percentage)], verbose=1)

    img_i = 0
    while True:
        fig, axes = plt.subplots(2, 3)
        for i in range(2):
            #ix = random.randint(0, len(preds_train) - 1)
            ix = (img_i + i) % len(X_test)

            # Plot the images in the subplots
            noisy_percentage = str((img_i//2)*2 + 1) + "%"
            axes[i][0].imshow(X_test[ix], cmap='gray')
            axes[i][0].set_title('noisy ' + noisy_percentage)

            axes[i][1].imshow(preds_train[ix], cmap='gray')
            axes[i][1].set_title('predict')

            axes[i][2].imshow(Y_test[ix], cmap='gray')
            axes[i][2].set_title('clean')

        img_i = (img_i + 2) % len(X_test)

        # Adjust spacing between subplots
        plt.tight_layout()

        # Display the plot
        plt.show()
