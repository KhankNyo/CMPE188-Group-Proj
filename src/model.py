import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_count, output_count):
    model = models.Sequential([ 
        # input
        tf.keras.Input(shape=(input_count,)),
        layers.Flatten(),

        # hidden
        layers.Dense(8),
        layers.Activation('relu'),

        layers.Dense(2),
        layers.Activation('relu'),

        # output
        layers.Dense(output_count, activation='relu', name='output')
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model
