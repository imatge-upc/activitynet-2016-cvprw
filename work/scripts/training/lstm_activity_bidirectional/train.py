import os
import pickle

import numpy as np

from work.dataset.activitynet import ActivityNetDataset
from work.environment import (DATASET_LABELS, DATASET_VIDEOS,
                              STORED_DATASET_PATH)
from work.models.decoder import BidirectionalLSTMModel


def train():
    nb_experiment = 1
    batch_size = 1
    timesteps = 500 # Entre 16 i 30
    epochs = 100
    output_size = 201
    hidden_units = 512

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

    print('Compiling model')
    model = BidirectionalLSTMModel(hidden_units, output_size, summary=True)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print('Model Compiled!')

    print('Loading data...')
    f_input = h5py.File(, 'r')
    f_output = h5py.File(, 'r')
    X = f_input['data']['training']
    Y = f_output['data']['training']

    print('Loading Data Finished!')

    for i in range(1, 1):
        print('Epoch {}/{}'.format(i, epochs))
        model.fit(X,
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
