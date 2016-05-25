import os
import sys

import h5py

from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop


def train():
    nb_experiment = 2
    batch_size = 256
    timesteps = 20 # Entre 16 i 30
    epochs = 100
    lr = 0.0001
    validation_samples = 5000

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
    # lstm = LSTM(512, return_sequences=True, name='lstm1')(input_normalized)
    lstm_classification = LSTM(512, return_sequences=False, name='lstm_classification')(input_dropout)
    lstm_detection = LSTM(512, return_sequences=True, name='lstm_detection')(input_dropout)
    dropout_classification = Dropout(p=.5)(lstm_classification)
    dropout_detection = Dropout(p=.5)(lstm_detection)

    classification = Dense(201, activation='softmax', name='class_predictor')(dropout_classification)
    detection = TimeDistributed(Dense(1, activation='sigmoid'), name='activity_detection')(dropout_detection)

    model = Model(input=input_features, output=[classification, detection])
    model.summary()
    rmsprop = RMSprop(lr=lr)
    loss = {
        'class_predictor': 'categorical_crossentropy',
        'activity_detection': 'binary_crossentropy'
    }
    model.compile(loss=loss, optimizer=rmsprop, metrics=['accuracy'])
    print('Model Compiled!')


    print('Loading Training Data...')
    f_dataset = h5py.File('../dataset/training_data_prediction.hdf5', 'r')
    X = f_dataset['training']['input_features']
    Y_classification = f_dataset['training']['classification_output']
    Y_detection = f_dataset['training']['detection_output']
    print('Loading Validation Data...')
    X_val = f_dataset['validation']['input_features'][:validation_samples,:,:]
    Y_classification_val = f_dataset['validation']['classification_output'][:validation_samples,:]
    Y_detection_val = f_dataset['validation']['detection_output'][:validation_samples,:,:]
    # print('Loading sample weights...')
    # sample_weights_classification = np.ones(Y_classification.shape)
    # sample_weights_detection = f_dataset['training']['sample_weights']
    print('Loading Data Finished!')
    print('Input shape: {}'.format(X.shape))
    print('Output classification shape: {}'.format(Y_classification.shape))
    print('Output detection shape: {}'.format(Y_detection.shape))
    print('Validation Input shape: {}'.format(X_val.shape))
    print('Validation Output classification shape: {}'.format(Y_classification_val.shape))
    print('Validation Output detection shape: {}'.format(Y_detection_val.shape))
    # print('Sample weights shape: {}\n'.format(sample_weights_detection.shape))

    for i in range(1, epochs+1):
        print('Epoch {}/{}'.format(i, epochs))
        model.fit(X,
                  {'class_predictor': Y_classification,
                  'activity_detection': Y_detection},
                  batch_size=batch_size,
                  validation_data=(X_val, {'class_predictor': Y_classification_val,
                                           'activity_detection': Y_detection_val}),
                #   sample_weight={'class_predictor': sample_weights_classification,
                #                 'activity_detection': sample_weights_detection},
                  verbose=1,
                  nb_epoch=1,
                  shuffle='batch')
        if (i % 5) == 0:
            print('Saving snapshot...')
            save_name = store_weights_file.format(nb_experiment=nb_experiment, epoch=i)
            save_path = os.path.join(store_weights_root, save_name)
            model.save_weights(save_path)

if __name__ == '__main__':
    train()
