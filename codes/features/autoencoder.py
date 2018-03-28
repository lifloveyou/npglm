import pickle
import numpy as np
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# dataset_path = 'citation/data/dataset_db.pkl'
# dataset = pickle.load(open(dataset_path, 'rb'))
# X_list, Y = dataset['X'], dataset['Y']


def encode(X_list, Y):
    for i in range(1, len(X_list)):
        X_list[i] -= X_list[i-1]
    scaler = MinMaxScaler(copy=False)
    for X in X_list:
        scaler.fit_transform(X)

    X = np.stack(X_list, axis=1) # X.shape = (n_samples, timesteps, n_features)
    n_samples, timesteps, input_dim = X.shape
    latent_dim = 2 * input_dim
    x_train, x_test = train_test_split(X, stratify=Y)

    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(latent_dim)(inputs)

    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(input_dim, return_sequences=True)(decoded)

    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    sequence_autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    sequence_autoencoder.fit(x_train, x_train, epochs=100, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
    # sequence_autoencoder.save('autorencoder.model')
    return encoder.predict(X)
