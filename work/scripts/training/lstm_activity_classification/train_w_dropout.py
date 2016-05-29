import os
import sys

import h5py

from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop


def train():
    nb_experiment = 9
    batch_size = 256
    timesteps = 20
    epochs = 200
    lr = 1e-5

    sys.stdout = open('./logs/training_e{:02d}.log'.format(nb_experiment), 'w')

    print('nb_experiment: {}'.format(nb_experiment))
    print('batch_size: {}'.format(batch_size))
    print('timesteps: {}'.format(timesteps))
    print('epochs: {}'.format(epochs))
    print('learning rate: {}\n'.format(lr))

    store_weights_root = './model_snapshot'
    store_weights_file = 'lstm_activity_classification_{nb_experiment:02d}_e{epoch:03}.hdf5'

    print('Compiling model')
    input_features = Input(batch_shape=(batch_size, timesteps, 4096,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(p=0.5)(input_normalized)
    lstm1 = LSTM(512, return_sequences=True, stateful=True, name='lstm1')(input_dropout)
    #lstm2 = LSTM(512, return_sequences=True, stateful=True, name='lstm2')(lstm1)
    output_dropout = Dropout(p=0.5)(lstm1)
    output = TimeDistributed(Dense(201, activation='softmax'), name='fc')(output_dropout)

    model = Model(input=input_features, output=output)
    model.summary()
    rmsprop = RMSprop(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'],
        sample_weight_mode='temporal')
    print('Model Compiled!')

    print('Loading Training Data...')
    f_dataset = h5py.File('../dataset/training_data_stateful.hdf5', 'r')
    X = f_dataset['X_features']
    Y = f_dataset['Y']
    print('Loading Validation Data...')
    X_val = f_dataset['X_features_val']
    Y_val = f_dataset['Y_val']
    print('Loading Sample Weights...')
    sample_weights = f_dataset['sample_weights'][...]
    sample_weights[sample_weights != 1] = .3
    print('Loading Data Finished!')
    print('Input shape: {}'.format(X.shape))
    print('Output shape: {}\n'.format(Y.shape))
    print('Validation Input shape: {}'.format(X_val.shape))
    print('Validation Output shape: {}'.format(Y_val.shape))
    print('Sample Weights shape: {}'.format(sample_weights.shape))

    for i in range(1, epochs+1):
        print('Epoch {}/{}'.format(i, epochs))
        model.fit(X,
                  Y,
                  batch_size=batch_size,
                  validation_data=(X_val, Y_val),
                  sample_weight=sample_weights,
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
