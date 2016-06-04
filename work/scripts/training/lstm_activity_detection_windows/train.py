import os
import sys

import h5py
import numpy as np

from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop


def train():
    nb_experiment = 1
    batch_size = 256
    timesteps = 20 # Entre 16 i 30
    epochs = 20
    lr = 1e-5
    #validation_samples = 5000

    video_features_size = 4096
    mfcc_size = 80
    spec_size = 8

    sys.stdout = open('./logs/training_e{:02d}.log'.format(nb_experiment), 'r+')

    print('nb_experiment: {}'.format(nb_experiment))
    print('batch_size: {}'.format(batch_size))
    print('timesteps: {}'.format(timesteps))
    print('epochs: {}'.format(epochs))
    print('learning rate: {}\n'.format(lr))

    store_weights_root = './model_snapshot'
    store_weights_file = 'lstm_activity_detection_windows_{nb_experiment:02d}_e{epoch:03}.hdf5'

    print('Compiling model')
    input_video_features = Input(batch_shape=(batch_size, timesteps, video_features_size,),
        name='video_features')
    video_normalized = BatchNormalization(name='video_features_normalization')(input_video_features)
    # input_mfcc_features = Input(batch_shape=(batch_size, timesteps, mfcc_size,),
    #     name='mfcc_features')
    # mfcc_normalized = BatchNormalization(name='mfcc_features_normalization')(input_mfcc_features)
    # input_spec_features = Input(batch_shape=(batch_size, timesteps, spec_size,),
    #     name='spec_features')
    # spec_normalized = BatchNormalization(name='spec_features_normalization')(input_spec_features)
    # input_merged_features = merge([video_normalized, mfcc_normalized, spec_normalized], mode='concat')
    input_dropout = Dropout(p=.5)(video_normalized)
    lstm = LSTM(512, return_sequences=False, stateful=True, name='lstm1')(input_dropout)

    detection = Dense(201, activation='sigmoid', name='activity_detection')(lstm)

    model = Model(input=input_video_features, output=detection)
    model.summary()
    rmsprop = RMSprop(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
    print('Model Compiled!')

    print('Loading Training Data...')
    f_dataset = h5py.File('../dataset/training_data_stateful.hdf5', 'r')
    X = f_dataset['X_features']
    Y = f_dataset['Y'][...]
    Y = np.mean(Y, axis=1)
    print('Loading Validation Data...')
    X_val = f_dataset['X_features_val']
    Y_val = f_dataset['Y_val'][...]
    Y_val = np.mean(Y_val, axis=1)
    print('Loading Data Finished!')
    print('Input shape: {}'.format(X.shape))
    print('Output detection shape: {}'.format(Y.shape))
    print('Validation Input shape: {}'.format(X_val.shape))
    print('Validation Output detection shape: {}'.format(Y_val.shape))

    for i in range(1, epochs+1):
        print('Epoch {}/{}'.format(i, epochs))
        model.fit(X,
                  Y,
                  batch_size=batch_size,
                  validation_data=(X_val, Y_val),
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
