import os
import sys

import h5py

from work.environment import FEATURES_DATASET_FILE, OUTPUTS_DATASET_FILE
from work.models.decoder import RecurrentFeedbackActivityDetectionNetwork
from work.processing.data import load_features_data_h5_feedback


def train():
    nb_experiment = 2
    batch_size = 256
    timesteps = 20 # Entre 16 i 30
    epochs = 100

    store_weights_root = './model_snapshot'
    store_weights_file = 'lstm_activity_classification_feedback_{nb_experiment:02d}_e{epoch:03}.hdf5'

    print('Compiling model')
    model = RecurrentFeedbackActivityDetectionNetwork(batch_size, timesteps, summary=True)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print('Model Compiled!')

    print('Loading Data...')
    f_input = h5py.File(FEATURES_DATASET_FILE, 'r')
    f_output = h5py.File(OUTPUTS_DATASET_FILE, 'r')
    X_features, X_output, Y = load_features_data_h5_feedback(f_input, f_output, timesteps, batch_size)
    print('Loading Data Finished!')
    print('Input features shape: {}'.format(X_features.shape))
    print('Input previous output shape: {}'.format(X_output.shape))
    print('Output shape: {}\n'.format(Y.shape))

    for i in range(1, epochs+1):
        print('Epoch {}/{}'.format(i, epochs))
        sys.stdout.flush()
        model.fit([X_features, X_output],
                  Y,
                  batch_size=batch_size,
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
