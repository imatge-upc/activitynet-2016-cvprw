import os
import random
import sys

import h5py
import numpy as np

from keras.callbacks import Callback
from keras.layers import LSTM, BatchNormalization, Dense, Input, TimeDistributed, merge
from keras.models import Model
from keras.optimizers import RMSprop


class AccuracyComputation(Callback):
    def __init__(self, model, num_samples=800):
        self.num_samples = num_samples
        self.f_audio_features = h5py.File('../dataset/stateful_dataset_with_audio.hdf5', 'r')
        self.f_output = h5py.File('/imatge/amontes/work/datases/ActivityNet/v1.3/')
        self.training_vid_id = random.sample(self.f_audio_features['training'].keys(), num_samples)
        self.validation_vid_id = random.sample(self.f_audio_features['validation'].keys(), num_samples)
        self.model = model
        self.accuracies = []

    def on_epoch_end(self, logs={}):
        # Computing training accuracy
        accuracy_vector = np.zeros((self.num_samples,))
        pos = 0
        for vid_id in self.training_vid_id:
            mfcc_features = self.f_audio_features['training']['mfcc'][vid_id][...]
            nb_clips = mfcc_features.shape[0]
            mfcc_features = mfcc_features.reshape((nb_clips, 1, 80))
            spec_features = self.f_audio_features['training']['spec'][vid_id][...]
            spec_features = spec_features.reshape(nb_clips, 1, 8)
            self.model.reset_states()
            prediction = self.model.predict({'mfcc_features': mfcc_features, 'spec_features': spec_features})
            prediction = prediction.reshape(nb_clips, 201)
            accuracy_vector[pos] = (np.argmax(prediction[1:]) + 1) == self.f_output['training'][vid_id]
            pos += 1

        # Computing validation accuracy
        val_accuracy_vector = np.zeros((self.num_samples,))
        pos = 0
        for vid_id in self.validation_vid_id:
            mfcc_features = self.f_audio_features['validation']['mfcc'][vid_id][...]
            nb_clips = mfcc_features.shape[0]
            mfcc_features = mfcc_features.reshape((nb_clips, 1, 80))
            spec_features = self.f_audio_features['validation']['spec'][vid_id][...]
            spec_features = spec_features.reshape(nb_clips, 1, 8)
            self.model.reset_states()
            prediction = self.model.predict({'mfcc_features': mfcc_features, 'spec_features': spec_features})
            prediction = prediction.reshape(nb_clips, 201)
            val_accuracy_vector[pos] = (np.argmax(prediction[1:]) + 1) == self.f_output['validation'][vid_id]
            pos += 1

        accuracy = accuracy_vector.mean()
        validation_accuracy = val_accuracy_vector.mean()

        print('Computing Accuracy: loss: {:.4f} - acc: {:.4f} - val_loss: {:.4f} - val_acc: {:.4f}'.format(
            logs.get('loss'), accuracy, logs.get('val_loss'), validation_accuracy))


def train():
    nb_experiment = 3
    batch_size = 256
    timesteps = 20
    epochs = 100
    lr = 1e-4
    background_weight = 0.6

    # validation_batches = 20
    mfcc_size = 80
    spec_size = 8

    sys.stdout = open('./logs/training_e{:02d}.log'.format(nb_experiment), 'w')

    print('nb_experiment: {}'.format(nb_experiment))
    print('batch_size: {}'.format(batch_size))
    print('timesteps: {}'.format(timesteps))
    print('epochs: {}'.format(epochs))
    print('learning rate: {}'.format(lr))
    print('background_weight: {}\n'.format(background_weight))

    store_weights_root = './model_snapshot'
    store_weights_file = 'lstm_activity_classification_{nb_experiment:02d}_e{epoch:03}.hdf5'

    print('Compiling model')
    input_mfcc_features = Input(batch_shape=(batch_size, timesteps, mfcc_size,),
        name='mfcc_features')
    mfcc_normalization = BatchNormalization(name='mfcc_normalization')
    input_mfcc_normalized = mfcc_normalization(input_mfcc_features)
    input_spec_features = Input(batch_shape=(batch_size, timesteps, spec_size,),
        name='spec_features')
    spec_normalization = BatchNormalization(name='spec_normalization')
    input_spec_normalized = spec_normalization(input_spec_features)
    input_merged = merge([input_mfcc_normalized, input_spec_normalized], mode='concat',
        concat_axis=-1)

    lstm = LSTM(512, return_sequences=True, stateful=True, name='lstm')
    lstm_output = lstm(input_merged)
    softmax = TimeDistributed(Dense(201, activation='softmax'), name='softmax')
    output = softmax(lstm_output)

    model = Model(input=[input_mfcc_features, input_spec_features], output=output)
    model.summary()
    rmsprop = RMSprop(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, sample_weight_mode='temporal')


    input_mfcc_features_val = Input(batch_shape=(1, 1, mfcc_size,),
        name='mfcc_features')
    input_mfcc_normalized_val = mfcc_normalization(input_mfcc_features_val)
    input_spec_features_val = Input(batch_shape=(1, 1, spec_size,),
        name='spec_features')
    input_spec_normalized_val = spec_normalization(input_spec_features_val)
    input_merged_val = merge([input_mfcc_normalized_val, input_spec_normalized_val], mode='concat',
        concat_axis=-1)

    lstm_output_val = lstm(input_merged_val)
    output_val = softmax(lstm_output_val)

    model_val = Model(input=[input_mfcc_features_val, input_spec_features_val], output=output_val)
    print('Model Compiled!')

    print('Loading Training Data...')
    f_dataset = h5py.File('../dataset/stateful_dataset_with_audio.hdf5', 'r')
    mfcc_features = f_dataset['training']['mfcc_features']
    spec_features = f_dataset['training']['spec_features']
    output = f_dataset['training']['output']
    # print('Loading Validation Data...')
    # mfcc_features_val = f_dataset['validation']['mfcc_features'][:validation_batches*batch_size]
    # spec_features_val = f_dataset['validation']['spec_features'][:validation_batches*batch_size]
    # output_val = f_dataset['validation']['output'][:validation_batches*batch_size]
    print('Loading Sample Weights...')
    sample_weight = f_dataset['training']['sample_weight'][...]
    sample_weight[sample_weight != 1] = background_weight
    print('Loading Data Finished!\n')

    print('MFCC features shape: {}'.format(mfcc_features.shape))
    print('Spec features shape: {}'.format(spec_features.shape))
    print('Output shape: {}'.format(output.shape))
    print('Sample Weight shape: {}\n'.format(sample_weight.shape))

    # print('Validation MFCC features shape: {}'.format(mfcc_features_val.shape))
    # print('Validation Spec features shape: {}'.format(spec_features_val.shape))
    # print('Validation Output shape: {}\n'.format(output_val.shape))

    accuracy_callback = AccuracyComputation(model_val)

    for i in range(1, epochs+1):
        print('Epoch {}/{}'.format(i, epochs))
        model.fit({'mfcc_features': mfcc_features, 'spec_features': spec_features},
                  output,
                  batch_size=batch_size,
                #   validation_data=({'mfcc_features': mfcc_features_val,
                #                     'spec_features': spec_features_val},
                #                     output_val),
                  sample_weight=sample_weight,
                  callbacks=[accuracy_callback],
                  verbose=0,
                  nb_epoch=1,
                  shuffle=False)
        print('Reseting model states')
        model.reset_states()
        if (i % 5) == 0:
            print('Saving snapshot...')
            save_name = store_weights_file.format(nb_experiment=nb_experiment, epoch=i)
            save_path = os.path.join(store_weights_root, save_name)
            model.save_weights(save_path)

    f_dataset.close()

if __name__ == '__main__':
    train()
