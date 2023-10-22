import tensorflow as tf
from argparse import ArgumentParser
import sys

# local modules
import data
import model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--model-name', type=str, default='unet_denoise.h5', help='model name')
    parser.add_argument('--save-best-model-only', type=bool, default=True, help='save the best model only')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    print('----------------- Denoise Images Using U-Net -----------------')
    print('Author: Gregorio Osorio')
    print('GitHub: gregorioosorio')
    print('--------------------------------------------------------------')

    print('----------------- GPU Setup -----------------')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=4*1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    print('----------------- Loading Data -----------------')
    X_train, Y_train = data.load_data()

    print('----------------- Building Model -----------------')
    model = model.u_net((256,256,1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss=tf.keras.losses.MeanSquaredError(), metrics=['mse'])
    model.summary()

    checkpointer = tf.keras.callbacks.ModelCheckpoint(args.model_name, verbose=1, save_best_only=args.save_best_model_only)
    callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
            tf.keras.callbacks.TensorBoard(log_dir='logs'),
            checkpointer]

    print('----------------- Training Model -----------------')
    results = model.fit(X_train, Y_train, validation_split=0.3, batch_size=2, epochs=25, callbacks=callbacks)
