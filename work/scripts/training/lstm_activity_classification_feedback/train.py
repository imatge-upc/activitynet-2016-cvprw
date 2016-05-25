import os
import sys

import h5py

from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed, merge
from keras.models import Model
from keras.optimizers import RMSprop


def train():
    nb_experiment = 2
    batch_size = 256
    timesteps = 20 # Entre 16 i 30
    epochs = 100
    lr = 1e-5

    sys.stdout = open('./logs/training_e{:02d}.log'.format(nb_experiment), 'w')

    print('nb_experiment: {}'.format(nb_experiment))
    print('batch_size: {}'.format(batch_size))
    print('timesteps: {}'.format(timesteps))
    print('epochs: {}'.format(epochs))
    print('learning rate: {}\n'.format(lr))

    store_weights_root = './model_snapshot'
    store_weights_file = 'lstm_activity_classification_feedback_{nb_experiment:02d}_e{epoch:03}.hdf5'

    print('Compiling model')
    input_features = Input(batch_shape=(batch_size, timesteps, 4096,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    previous_output = Input(batch_shape=(batch_size, timesteps, 202,), name='prev_output')
    merging = merge([input_normalized, previous_output], mode='concat', concat_axis=-1)
    merging_dropout = Dropout(p=0.5)(merging)
    lstm1 = LSTM(512, return_sequences=True, stateful=True, name='lstm1')(merging_dropout)
    # lstm2 = LSTM(512, return_sequences=True, stateful=True, name='lstm2')(lstm1)
    output_dropout = Dropout(p=0.5)(lstm1)
    output = TimeDistributed(Dense(201, activation='softmax'), name='fc')(output_dropout)
    model = Model(input=[input_features, previous_output], output=output)

    model.summary()
    rmsprop = RMSprop(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
    print('Model Compiled!')

    print('Loading Training Data...')
    f_dataset = h5py.File('../dataset/training_data_stateful.hdf5', 'r')
    X_features = f_dataset['X_features']
    X_prev_output = f_dataset['X_prev_output']
    Y = f_dataset['Y']
    print('Loading Validation Data...')
    X_features_val = f_dataset['X_features_val']
    X_prev_output_val = f_dataset['X_prev_output_val']
    Y_val = f_dataset['Y_val']
    print('Loading Data Finished!')
    print('Input shape: {}'.format(X_features.shape))
    print('Input previous output shape: {}'.format(X_prev_output.shape))
    print('Output shape: {}\n'.format(Y.shape))
    print('Validation Input shape: {}'.format(X_features_val.shape))
    print('Validation Input previous output shape: {}'.format(X_prev_output_val.shape))
    print('Validation Output shape: {}\n'.format(Y_val.shape))

    for i in range(1, epochs+1):
        print('Epoch {}/{}'.format(i, epochs))
        model.fit([X_features, X_prev_output],
                  Y,
                  batch_size=batch_size,
                  validation_data=([X_features_val, X_prev_output_val], Y_val),
                  verbose=1,
                  nb_epoch=1,
                  shuffle=False)
        print('Reseting model states')
        model.reset_states()
        if (i % 5) == 0:
            print('Saving snapshot...')
            save_name = store_weights_file.format(nb_experiment=nb_experiment, epoch=i)
            save_path = os.path.join(store_weights_root, save_name)
            model.save_weights(save_path)

if __name__ == '__main__':
    train()
