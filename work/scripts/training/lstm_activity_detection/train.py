import os
import sys

import h5py
import numpy as np

from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop


def train():
    nb_experiment = 6
    batch_size = 256
    timesteps = 20 # Entre 16 i 30
    epochs = 100
    lr = 1e-6
    #validation_samples = 5000

    sys.stdout = open('./logs/training_e{:02d}.log'.format(nb_experiment), 'w')

    print('nb_experiment: {}'.format(nb_experiment))
    print('batch_size: {}'.format(batch_size))
    print('timesteps: {}'.format(timesteps))
    print('epochs: {}'.format(epochs))
    print('learning rate: {}\n'.format(lr))

    store_weights_root = './model_snapshot'
    store_weights_file = 'lstm_activity_detection_{nb_experiment:02d}_e{epoch:03}.hdf5'

    print('Compiling model')
    input_features = Input(batch_shape=(batch_size, timesteps, 4096,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(p=.5)(input_normalized)
    lstm = LSTM(256, return_sequences=True, stateful=True, name='lstm1')(input_dropout)
    # dropout_detection = Dropout(p=.5)(lstm)

    detection = TimeDistributed(Dense(1, activation='sigmoid'), name='activity_detection')(lstm)

    model = Model(input=input_features, output=detection)
    model.summary()
    rmsprop = RMSprop(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
    print('Model Compiled!')


    print('Loading Training Data...')
    f_dataset = h5py.File('../dataset/training_data_stateful.hdf5', 'r')
    X = f_dataset['X_features']
    Y = f_dataset['Y'][...]
    nb_seq = Y.shape[0]
    Y_activity = np.zeros((nb_seq, 20, 1))
    Y_activity[Y[:,:,0] != 0] = 1
    print('Loading Validation Data...')
    X_val = f_dataset['X_features_val']
    Y_val = f_dataset['Y_val'][...]
    nb_seq_val = Y_val.shape[0]
    Y_activity_val = np.zeros((nb_seq_val, 20, 1))
    Y_activity_val[Y_val[:,:,0] != 0] = 1
    print('Loading Data Finished!')
    print('Input shape: {}'.format(X.shape))
    print('Output detection shape: {}'.format(Y_activity.shape))
    print('Validation Input shape: {}'.format(X_val.shape))
    print('Validation Output detection shape: {}'.format(Y_activity_val.shape))

    for i in range(1, epochs+1):
        print('Epoch {}/{}'.format(i, epochs))
        model.fit(X,
                  Y_activity,
                  batch_size=batch_size,
                  validation_data=(X_val, Y_activity_val),
                  verbose=1,
                  nb_epoch=1,
                  shuffle='batch')
        model.reset_states()
        if (i % 5) == 0:
            print('Saving snapshot...')
            save_name = store_weights_file.format(nb_experiment=nb_experiment, epoch=i)
            save_path = os.path.join(store_weights_root, save_name)
            model.save_weights(save_path)

if __name__ == '__main__':
    train()
