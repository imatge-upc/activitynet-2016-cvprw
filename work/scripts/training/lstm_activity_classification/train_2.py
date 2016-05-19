import os
import pickle

import h5py
import numpy as np

from keras import backend as K
from keras.objectives import categorical_crossentropy
from work.dataset.activitynet import ActivityNetDataset
from work.environment import (ACTIVITY_CLASSIFICATION, DATASET_LABELS,
                              DATASET_VIDEOS, FEATURES_DATASET_FILE,
                              OUTPUTS_DATASET_FILE, STORED_DATASET_PATH)
from work.models.decoder import RecurrentFeedbackActivityDetectionNetwork
from work.processing.data import load_features_data_h5_feedback


def weighted_categorical_crossentropy(weights):
    def custom_loss(y_true, y_pred):
        score_array = categorical_crossentropy(y_true, y_pred)
        score_array = K.mean(score_array, axis=0)
        score_array *= weights
        score_array /= K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
        return score_array

    return custom_loss


def train():
    nb_experiment = 2
    batch_size = 256
    timesteps = 20 # Entre 16 i 30
    epochs = 100
    output_size = 201

    store_weights_root = ACTIVITY_CLASSIFICATION['training_model_weights']
    store_weights_file = ACTIVITY_CLASSIFICATION['weights_file_pattern']

    # Loading dataset
    print('Loading dataset')
    if os.path.exists(STORED_DATASET_PATH):
        with open(STORED_DATASET_PATH, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = ActivityNetDataset(
            videos_path=DATASET_VIDEOS,
            labels_path=DATASET_LABELS
        )
        print('Generating Video Instances')
        dataset.generate_instances(length=16, overlap=0, subsets=('training',))
        with open(STORED_DATASET_PATH, 'wb') as f:
            pickle.dump(dataset, f)

    f_input = h5py.File(FEATURES_DATASET_FILE, 'r')
    f_output = h5py.File(OUTPUTS_DATASET_FILE, 'r')

    nb_videos = len(f_input['training'].keys())
    print('Number of videos: %d' % nb_videos)

    print('Get weights')
    weights = np.asarray([1 for _ in range(output_size)])
    weights[0] = 0.6

    print('Compiling model')
    model = RecurrentFeedbackActivityDetectionNetwork(batch_size, timesteps, summary=True)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print('Model Compiled!')

    print('Loading Generator...')
    X_features, X_output, Y = load_features_data_h5_feedback(f_input, f_output, timesteps, batch_size)
    print('Loading Data Finished!')

    for i in range(1, epochs+1):
        print('Epoch {}/{}'.format(i, epochs))
        model.fit([X_features, X_output],
                  Y,
                  batch_size=batch_size,
                  verbose=1,
                  nb_epoch=1,
                  validation_split=.1,
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
