import os
import sys

import h5py

from keras.layers import LSTM, BatchNormalization, Dense, Input, TimeDistributed, Dropout
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import Callback


class AccuracyComputation(Callback):

    def __init__(self, model, validation_data):
        self.model = model
        self.input, self.output = validation_data
        self.accuracies = []

    def on_epoch_end(self, logs={}):
        for val_features in self.input:
            predictions = model.predict(val_features)
            

def train():
    nb_experiment = 10
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
    store_weights_file = 'lstm_activity_classification_{nb_experiment:02d}_e{epoch:03}.hdf5'

    print('Compiling model')
    input_features = Input(batch_shape=(batch_size, timesteps, 4096,), name='features')
    normalization = BatchNormalization(name='normalization')
    input_normalized = normalization(input_features)
    input_dropout = Dropout(p=.5)(input_normalized)
    lstm = LSTM(512, return_sequences=True, stateful=True, name='lstm1')
    lstm_output = lstm(input_dropout)
    lstm_dropout = Dropout(p=.5)(lstm_output)
    softmax = TimeDistributed(Dense(201, activation='softmax'), name='fc')
    output = softmax(lstm_dropout)

    model = Model(input=input_features, output=output)
    model.summary()
    rmsprop = RMSprop(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

    # validation model
    input_features_val = Input(batch_size=(1, 1, 4096,), name='features_val')
    input_normalized_val = normalization(input_features_val)
    input_dropout_val = Dropout(p=.5)(input_normalized_val)
    lstm_output_val = lstm(input_dropout_val)
    lstm_dropout = Dropout(p=.5)(lstm_output_val)
    output_val = softmax(lstm_dropout)
    model_val = Model(input=input_features_val, output=output_val)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.summary()
    print('Model Compiled!')

    print('Loading Training Data...')
    f_dataset = h5py.File('../dataset/training_data_stateful.hdf5', 'r')
    X = f_dataset['X_features']
    Y = f_dataset['Y']
    print('Loading Validation Data...')
    X_val = f_dataset['X_features_val']
    Y_val = f_dataset['Y_val']
    print('Loading Data Finished!')
    print('Input shape: {}'.format(X.shape))
    print('Output shape: {}\n'.format(Y.shape))
    print('Validation Input shape: {}'.format(X_val.shape))
    print('Validation Output shape: {}\n'.format(Y_val.shape))

    for i in range(1, epochs+1):
        print('Epoch {}/{}'.format(i, epochs))
        model.fit(X,
                  Y,
                  batch_size=batch_size,
                  validation_data=(X_val, Y_val),
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
