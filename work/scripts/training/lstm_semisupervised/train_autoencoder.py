import os
import sys

import h5py

from keras.layers import (LSTM, BatchNormalization, Dense, Input, RepeatVector, TimeDistributed,
                          merge)
from keras.models import Model
from keras.optimizers import RMSprop


def train():
    nb_experiment = 4
    batch_size = 256
    timesteps = 20
    epochs = 100
    lr = 1e-5

    validation_batches = 30
    video_size = 4096
    mfcc_size = 80
    spec_size = 8
    nb_classes = 201

    sys.stdout = open('./logs/training_encoder_e{:02d}.log'.format(nb_experiment), 'w')

    print('nb_experiment: {}'.format(nb_experiment))
    print('batch_size: {}'.format(batch_size))
    print('timesteps: {}'.format(timesteps))
    print('epochs: {}'.format(epochs))
    print('learning rate: {}'.format(lr))

    store_weights_root = './model_snapshot'
    store_weights_file = 'lstm_encoder_{nb_experiment:02d}_e{epoch:03}.hdf5'
    model_architecture_file = './model_architecture/model_architecture_encoder_{experiment:02d}.yaml'.format(experiment=nb_experiment)

    print('Compiling model')
    input_ = Input(batch_shape=(batch_size, timesteps, 201,),
        name='output_as_input')

    lstm_encoder = LSTM(512, return_sequences=False, stateful=True, name='lstm')(input_)
    encoded = RepeatVector(timesteps)(lstm_encoder)
    lstm_decoder = LSTM(512, return_sequences=True, stateful=True, name='lstm_decoder')(encoded)

    output = TimeDistributed(Dense(201, activation='softmax'), name='output')(lstm_decoder)

    model = Model(
        input=input_,
        output=output
    )

    model.summary()
    with open(model_architecture_file, 'w') as f:
        f.write(model.to_yaml())
    rmsprop = RMSprop(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
    print('Model Compiled!')


    print('Loading Training Data...')
    f_dataset = h5py.File('../dataset/stateful_dataset_with_audio_feedback.hdf5', 'r')
    data = f_dataset['training']['output']
    print('Loading Validation Data...')
    data_val = f_dataset['validation']['output'][:validation_batches*batch_size]
    print('Loading Data Finished!\n')

    print('Output shape: {}'.format(data.shape))

    print('Validation Output shape: {}\n'.format(data_val.shape))

    for i in range(1, epochs+1):
        print('Epoch {}/{}'.format(i, epochs))
        model.fit(data,
                  data,
                  batch_size=batch_size,
                  validation_data=(data_val,data_val),
                  verbose=1,
                  nb_epoch=1,
                  shuffle=False)
        model.reset_states()
        if (i % 5) == 0:
            print('Saving snapshot...')
            save_name = store_weights_file.format(nb_experiment=nb_experiment, epoch=i)
            save_path = os.path.join(store_weights_root, save_name)
            model.save_weights(save_path)

if __name__ == '__main__':
    train()
