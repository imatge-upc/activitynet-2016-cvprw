import os
import pickle
import sys

from work.dataset.activitynet import ActivityNetDataset
from work.environment import (ACTIVITY_DETECTION, DATASET_LABELS,
                              DATASET_VIDEOS, STORED_DATASET_PATH,
                              STORED_FEATURES_PATH, STORED_MODELS_PATH)
from work.models.decoder import RecurrentBinaryActivityDetectionNetwork
from work.processing.data import load_features_data
from work.tools.utils import get_files_in_dir


def train():
    nb_experiment = 1
    batch_size = 256
    timesteps = 20 # Entre 16 i 30
    epochs = 100

    store_weights_root = ACTIVITY_DETECTION['training_model_weights']
    store_weights_file = ACTIVITY_DETECTION['weights_file_pattern']


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

    extracted_features = get_files_in_dir(STORED_FEATURES_PATH, extension='npy')
    # Remove the videos which features hasn't been extracted
    videos_to_remove = []
    for video in dataset.videos:
        if video.video_id not in extracted_features:
            videos_to_remove.append(video)
    for video in videos_to_remove:
        dataset.videos.remove(video)

    videos = dataset.get_subset_videos('training')
    nb_videos = len(dataset.videos)
    print('Number of videos: %d' % nb_videos)
    nb_instances = len(dataset.instances_training)
    print('Number of instances: %d' % nb_instances)
    sys.stdout.flush()

    print('Compiling model')
    model = RecurrentBinaryActivityDetectionNetwork(batch_size, timesteps, summary=True)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print('Model Compiled!')

    print('Loading Data...')
    X, Y = load_features_data(videos, timesteps, batch_size, output_mode='binary_activity')
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
            save_name = store_weights_file.format(nb_experiment=nb_experiment, epoch=i)
            save_path = os.path.join(store_weights_root, save_name)
            model.save_weights(save_path)

if __name__ == '__main__':
    train()
