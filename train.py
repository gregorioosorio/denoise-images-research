import tensorflow as tf
from argparse import ArgumentParser
import sys

# local modules
import data
import model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--learning-rate', type=float, default=1e-6, help='learning rate')
    parser.add_argument('--model-name', type=str, default='unet_denoise.h5', help='model name')
    parser.add_argument('--save-best-model-only', type=bool, default=True, help='save the best model only')
    parser.add_argument('--data-augmentation', type=bool, default=True, help='apply data augmentation')
    parser.add_argument('--batch-size', type=int, default=16, help='fit batch size')
    parser.add_argument('--gpu-ram', type=int, default=16384, help='gpu ram MB')

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
            # Restrict TensorFlow to only allocate args.gpu_ram GB of memory on the first GPU
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=args.gpu_ram)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    print('----------------- Loading Data -----------------')
    X_train, Y_train = data.load_data(data_augmentation=args.data_augmentation)

    print('----------------- Building Model -----------------')
    model = model.u_net_gn((256,256,1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss=tf.keras.losses.MeanAbsoluteError(), metrics=['mae'])
    model.summary()

    checkpointer = tf.keras.callbacks.ModelCheckpoint(args.model_name, verbose=1, save_best_only=args.save_best_model_only)
    callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
            tf.keras.callbacks.TensorBoard(log_dir='logs'),
            checkpointer]

    print('----------------- Training Model -----------------')
    results = model.fit(X_train, Y_train, validation_split=0.3, batch_size=args.batch_size, epochs=30, callbacks=callbacks)
