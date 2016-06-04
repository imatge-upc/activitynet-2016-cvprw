import os
import sys

import h5py

from keras.layers import BatchNormalization, Dense, Input, TimeDistributed, merge
from keras.models import Model
from keras.optimizers import SGD


def train():
    nb_experiment = 1
    batch_size = 256
    timesteps = 20
    epochs = 120
    lr = 0.1
    background_weight = 0.3

    validation_batches = 20
    video_size = 4096
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
    store_weights_file = 'lstm_activity_classification_01_e{epoch:03d}.hdf5'

    input_video_features = Input(batch_shape=(batch_size, timesteps, video_size,),
        name='video_features')
    input_video_normalized = BatchNormalization(name='video_normalization')(input_video_features)
    input_mfcc_features = Input(batch_shape=(batch_size, timesteps, mfcc_size,),
        name='mfcc_features')
    input_mfcc_normalized = BatchNormalization(name='mfcc_normalization')(input_mfcc_features)
    input_spec_features = Input(batch_shape=(batch_size, timesteps, spec_size,),
        name='spec_features')
    input_spec_normalized = BatchNormalization(name='spec_normalization')(input_spec_features)
    input_merged = merge([input_video_normalized, input_mfcc_normalized, input_spec_normalized],
        mode='concat', concat_axis=-1)

    hidden_layer = TimeDistributed(Dense(1024, activation='softmax'), name='hidden')(input_merged)
    output = TimeDistributed(Dense(201, activation='softmax'), name='output')(hidden_layer)

    model = Model(input=[input_video_features, input_mfcc_features, input_spec_features],
        output=output)
    model.summary()
    sgd = SGD(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'],
        sample_weight_mode='temporal')
    print('Model Compiled!')

    print('Loading Training Data...')
    f_dataset = h5py.File('../dataset/stateful_dataset_with_audio.hdf5', 'r')
    video_features = f_dataset['training']['vid_features']
    mfcc_features = f_dataset['training']['mfcc_features']
    spec_features = f_dataset['training']['spec_features']
    output = f_dataset['training']['output']
    print('Loading Validation Data...')
    video_features_val = f_dataset['validation']['vid_features'][:validation_batches*batch_size]
    mfcc_features_val = f_dataset['validation']['mfcc_features'][:validation_batches*batch_size]
    spec_features_val = f_dataset['validation']['spec_features'][:validation_batches*batch_size]
    output_val = f_dataset['validation']['output'][:validation_batches*batch_size]
    print('Loading Sample Weights...')
    sample_weight = f_dataset['training']['sample_weight'][...]
    sample_weight[sample_weight != 1] = background_weight
    print('Loading Data Finished!\n')

    print('MFCC features shape: {}'.format(mfcc_features.shape))
    print('Spec features shape: {}'.format(spec_features.shape))
    print('Output shape: {}'.format(output.shape))
    print('Sample Weight shape: {}\n'.format(sample_weight.shape))

    print('Validation MFCC features shape: {}'.format(mfcc_features_val.shape))
    print('Validation Spec features shape: {}'.format(spec_features_val.shape))
    print('Validation Output shape: {}\n'.format(output_val.shape))

    for i in range(1, epochs+1):
        print('Epoch {}/{}'.format(i, epochs))
        model.fit({'video_features': video_features, 'mfcc_features': mfcc_features,
                    'spec_features': spec_features},
                  output,
                  batch_size=batch_size,
                  validation_data=({'video_features': video_features_val,
                                    'mfcc_features': mfcc_features_val,
                                    'spec_features': spec_features_val},
                                    output_val),
                  sample_weight=sample_weight,
                  verbose=1,
                  nb_epoch=1,
                  shuffle='batch')
        if (i % 5) == 0:
            print('Saving snapshot...')
            save_name = store_weights_file.format(nb_experiment=nb_experiment, epoch=i)
            save_path = os.path.join(store_weights_root, save_name)
            model.save_weights(save_path)

    f_dataset.close()

if __name__ == '__main__':
    train()
