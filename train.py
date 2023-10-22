import tensorflow as tf
from argparse import ArgumentParser
import sys

# local modules
import data
import model

if __name__ == '__main__':
    parser = ArgumentParser()
    # TODO: add arguments to be parsed

    if __name__ == '__main__':
        parser = ArgumentParser()
        # TODO: add arguments to be parsed

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

        print('----------------- Building Model -----------------')
        model = model.u_net((256,256,1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-9), loss=tf.keras.losses.MeanSquaredError(), metrics=['mse'])
        model.summary()

        checkpointer = tf.keras.callbacks.ModelCheckpoint('model1_for_denoise.h5', verbose=1, save_best_only=True)
        callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
                tf.keras.callbacks.TensorBoard(log_dir='logs')]

        print('----------------- Training Model -----------------')
        results = model.fit(X_train, Y_train, validation_split=0.3, batch_size=16, epochs=25, callbacks=callbacks)

