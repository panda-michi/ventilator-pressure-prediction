from tensorflow import keras

def load_bidLSTM(input_dim, hidden0, hidden1, hidden2, hidden3, dense):
    model = keras.models.Sequential([
            keras.layers.Input(shape = (80, input_dim)),
            keras.layers.Bidirectional(keras.layers.LSTM(hidden0, return_sequences = True)),
            keras.layers.Bidirectional(keras.layers.LSTM(hidden1, return_sequences = True)),
            keras.layers.Bidirectional(keras.layers.LSTM(hidden2, return_sequences = True)),
            keras.layers.Bidirectional(keras.layers.LSTM(hidden3, return_sequences = True)),
            keras.layers.Dense(dense, activation = 'selu'),
            keras.layers.Dense(1),
        ])

    return model

# referenced from https://www.kaggle.com/dlaststark/gb-vpp-pulp-fiction
def load_dlast(input_dim):
    x_input = keras.layers.Input(shape=(80, input_dim))
    
    x1 = keras.layers.Bidirectional(keras.layers.LSTM(units=768, return_sequences=True))(x_input)
    x2 = keras.layers.Bidirectional(keras.layers.LSTM(units=512, return_sequences=True))(x1)
    x3 = keras.layers.Bidirectional(keras.layers.LSTM(units=384, return_sequences=True))(x2)
    x4 = keras.layers.Bidirectional(keras.layers.LSTM(units=256, return_sequences=True))(x3)
    x5 = keras.layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=True))(x4)
    
    z2 = keras.layers.Bidirectional(keras.layers.GRU(units=384, return_sequences=True))(x2)
    
    z31 = keras.layers.Multiply()([x3, z2])
    z31 = keras.layers.BatchNormalization()(z31)
    z3 = keras.layers.Bidirectional(keras.layers.GRU(units=256, return_sequences=True))(z31)
    
    z41 = keras.layers.Multiply()([x4, z3])
    z41 = keras.layers.BatchNormalization()(z41)
    z4 = keras.layers.Bidirectional(keras.layers.GRU(units=128, return_sequences=True))(z41)
    
    z51 = keras.layers.Multiply()([x5, z4])
    z51 = keras.layers.BatchNormalization()(z51)
    z5 = keras.layers.Bidirectional(keras.layers.GRU(units=64, return_sequences=True))(z51)
    
    x = keras.layers.Concatenate(axis=2)([x5, z2, z3, z4, z5])
    
    x = keras.layers.Dense(units=128, activation='selu')(x)
    
    x_output = keras.layers.Dense(units=1)(x)

    model = tf.keras.Model(inputs=x_input, outputs=x_output, name='dlast_model')

    return model