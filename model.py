import tensorflow as tf

def down_sampling(x, filters, kernel_size=3, activation='relu', dropout_rate=0.1, use_maxpool = True):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same')(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same')(x)
    if use_maxpool:
        return tf.keras.layers.MaxPooling2D(pool_size=2)(x), x
    else:
        return x

def up_sampling(x, y, filters, activation='relu',dropout_rate=0.2):
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding='same')(x)
    x = tf.keras.layers.concatenate([x, y])
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, activation=activation, padding='same')(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, activation=activation, padding='same')(x)
    return x

def u_net(input_size = (256,256,1)):
    filters = [64, 128, 256, 512, 1024]

    # encoder
    inputs = tf.keras.layers.Input(input_size)
    x, temp1 = down_sampling(inputs, filters[0], dropout_rate=0.1)
    x, temp2 = down_sampling(x, filters[1], dropout_rate=0.1)
    x, temp3 = down_sampling(x, filters[2], dropout_rate=0.2)
    x, temp4 = down_sampling(x, filters[3], dropout_rate=0.2)
    x = down_sampling(x, filters[4], use_maxpool=False, dropout_rate=0.3)

    # decoder
    x = up_sampling(x, temp4, filters[3], dropout_rate=0.2)
    x = up_sampling(x, temp3, filters[2], dropout_rate=0.2)
    x = up_sampling(x, temp2, filters[1], dropout_rate=0.1)
    x = up_sampling(x, temp1, filters[0], dropout_rate=0.1)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name='u_net')

    return model

if __name__ == '__main__':
    model = u_net((256,256,1))
    model.summary()