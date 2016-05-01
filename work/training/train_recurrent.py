import os
import random

import numpy as np

from keras.utils import np_utils
from work.config import STORED_FEATURES_PATH, STORED_MODELS_PATH
from work.dataset.activitynet import ActivityNetDataset
from work.models.decoder import RecurrentNetwork


def load_data(videos, timesteps, batch_size):
    length = 16
    features_size = 4096
    output_size = 201

    random.shuffle(videos)

    stacks = []
    for _ in range(batch_size):
        stacks.append([])
    stacks_size = np.zeros(batch_size)

    for video in videos:
        pos = np.argmin(stacks_size)
        stacks[pos] += [video]
        stacks_size[pos] += video.num_frames // length

    max_seq = np.max(stacks_size)
    min_seq = np.min(stacks_size)

    data = np.zeros((batch_size, max_seq, features_size))
    output = np.zeros((batch_size, max_seq, output_size))
    for i in range(batch_size):
        pos = 0
        output_clases = []
        for video in stacks[i]:
            features_path = STORED_FEATURES_PATH + '/' + video_id + '.npy'
            video_features = np.load(features_path)
            assert video_features.size[1] == features_size
            nb_video_instances = video.num_frames // length
            assert video_features.size[0] == nb_video_instances
            data[i, pos:pos+nb_video_instances, :] = video_features
            pos += nb_video_instances
            for instance in video.instances:
                output_clases.append(instance.output)
        output[i,:,:] = np_utils.to_categorical(output_clases, nb_classes=output_size)

    nb_batches = min_seq // timesteps

    data = data[:,:nb_batches*timesteps,:]
    data = data.reshape((batch_size, nb_batches, timesteps, features_size))
    data = data.transpose(1, 0, 2, 3)

    output = output[:,:nb_batches*timesteps,:]
    output = output.reshape((batch_size, nb_batches, timesteps, features_size))
    output = output.transpose(1, 0, 2, 3)

    return data, output

def train():
    nb_experiment = 1
    batch_size = 256
    timesteps = 20 # Entre 16 i 30
    epochs = 100
    length = 16
# Idea a util function to check the log dirs and get the number of running it is ang generate all the login on a new file. Maybe do it through a bash action and pass it as an argument to the main function
# train_recurrent $(ls -l logs/ | grep train_recurrent | lc -l)
    # Loading dataset
    print('Loading dataset')
    dataset = ActivityNetDataset(
        videos_path='../../dataset/videos.json',
        labels_path='../../dataset/labels.txt'
    )
    print('Generating Video Instances')
    dataset.generate_instances(length=16, overlap=0)
    videos = dataset.get_subset_videos('training')
    nb_instances = sum([video.num_frames // length for video in dataset.videos])
    print('Number of instances: %d' % nb_instances)

    print('Generating classes weigths')
    class_weights = dataset.compute_class_weights()

    print('Compiling model')
    model = RecurrentNetwork(batch_size, timesteps, summary=True)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print('Model Compiled!')

    for i in range(epochs):
        print('Epoch {}/{}'.format(i, epochs))
        X, Y = load_data(videos, timesteps, batch_size)
        model.fit(X,
            Y,
            batch_size=batch_size,
            verbose=1,
            nb_epoch=1,
            shuffle=False,
            class_weights=class_weights
        )
        print('Reseting model states')
        model.reset_states()
        print('Saving snapshot...')
        save_name = 'recurrent_decoder_{nb_experiment:02d}_e{epoch:02}.hdf5'.format(nb_experiment=nb_experiment, epoch=i)
        save_path = os.path.join(STORED_MODELS_PATH, 'recurrent', save_name)
        model.save_weights(save_path)

if __name__ == '__main__':
    train()
