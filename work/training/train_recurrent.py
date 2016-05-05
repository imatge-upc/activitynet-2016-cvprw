import os
import pickle
import random
import sys

import numpy as np
from progressbar import ProgressBar

from keras.utils import np_utils
from work.config import (STORED_DATASET_PATH, STORED_FEATURES_PATH,
                         STORED_MODELS_PATH)
from work.dataset.activitynet import ActivityNetDataset
from work.models.decoder import RecurrentNetwork


def load_data(videos, timesteps, batch_size):
    length = 16
    features_size = 4096
    output_size = 201

    random.shuffle(videos)

    nb_instances = sum([video.num_frames // length for video in videos])

    data = np.zeros((nb_instances, features_size))
    output = np.zeros((nb_instances, output_size))
    pos = 0
    progbar = ProgressBar(max_value=nb_instances)
    for video in videos:
        features_path = STORED_FEATURES_PATH + '/' + video.video_id + '.npy'
        video_features = np.load(features_path)
        assert video_features.shape[1] == features_size
        nb_video_instances = len(video.instances)
        assert video_features.shape[0] == nb_video_instances, str(video_features.shape) + ' ' + str(nb_video_instances)
        data[pos:pos+nb_video_instances,:] = video_features
        output_clases = []
        for instance in video.instances:
            output_clases.append(instance.output)
        output[pos:pos+nb_video_instances,:] = np_utils.to_categorical(output_clases, nb_classes=output_size)
        pos += nb_video_instances
        progbar.update(pos)

    progbar.finish()
    assert pos == nb_instances

    nb_batches = (nb_instances // (timesteps * batch_size))
    total_length = nb_batches * batch_size * timesteps
    data, output = data[:total_length,:], output[:total_length,:]

    data = data.reshape((batch_size, nb_batches, timesteps, features_size))
    data = data.transpose(1, 0, 2, 3)
    data = data.reshape((nb_batches*batch_size, timesteps, features_size))
    output = output.reshape((batch_size, nb_batches, timesteps, output_size))
    output = output.transpose(1, 0, 2, 3)
    output = output.reshape((nb_batches*batch_size, timesteps, output_size))

    return data, output

def train():
    nb_experiment = 1
    batch_size = 256
    timesteps = 20 # Entre 16 i 30
    epochs = 100

    # Loading dataset
    print('Loading dataset')
    if os.path.exists(STORED_DATASET_PATH):
        with open(STORED_DATASET_PATH, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = ActivityNetDataset(
            videos_path='../../dataset/videos.json',
            labels_path='../../dataset/labels.txt'
        )
        print('Generating Video Instances')
        dataset.generate_instances(length=16, overlap=0, subsets=('training',))
        with open(STORED_DATASET_PATH, 'wb') as f:
            pickle.dump(dataset, f)
    videos = dataset.get_subset_videos('training')
    nb_instances = len(dataset.instances_training)
    print('Number of instances: %d' % nb_instances)
    sys.stdout.flush()

    print('Generating classes weigths')
    class_weights = {0: 0.6}
    for i in range(1, 201):
        class_weights.update({i: 1.})

    print('Compiling model')
    model = RecurrentNetwork(batch_size, timesteps, summary=True)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print('Model Compiled!')

    print('Loading Data...')
    X, Y = load_data(videos, timesteps, batch_size)
    print('Loading Data Finished!')
    print('Input shape: {}'.format(X.shape))
    print('Output shape: {}\n'.format(Y.shape))

    for i in range(1, epochs+1):
        print('Epoch {}/{}'.format(i, epochs))
        sys.stdout.flush()
        model.fit(X,
                  Y,
                  batch_size=batch_size,
                  verbose=1,
                  nb_epoch=1,
                  shuffle=False)
        print('Reseting model states')
        model.reset_states()
        if (i % 5) == 0:
            print('Saving snapshot...')
            save_name = 'recurrent_decoder_{nb_experiment:02d}_e{epoch:02}.hdf5'.format(nb_experiment=nb_experiment, epoch=i)
            save_path = os.path.join(STORED_MODELS_PATH, save_name)
            model.save_weights(save_path)

if __name__ == '__main__':
    train()
