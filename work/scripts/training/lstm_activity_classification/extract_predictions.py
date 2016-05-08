import sys

import numpy as np

from work.dataset.activitynet import ActivityNetDataset
from work.environment import (ACTIVITY_CLASSIFICATION, DATASET_LABELS,
                              DATASET_VIDEOS, STORED_FEATURES_PATH)
from work.models.decoder import RecurrentActivityClassificationNetwork
from work.tools.utils import get_files_in_dir


def extract_predicted_outputs():
    batch_size = 1
    timesteps = 1
    output_shape = 201

    print('Loading dataset')
    dataset = ActivityNetDataset(
        videos_path=DATASET_VIDEOS,
        labels_path=DATASET_LABELS
    )
    extracted_features = get_files_in_dir(STORED_FEATURES_PATH, extension='npy')
    # Remove the videos which features hasn't been extracted
    videos_to_remove = []
    for video in dataset.videos:
        if video.video_id not in extracted_features:
            videos_to_remove.append(video)
    for video in videos_to_remove:
        dataset.videos.remove(video)

    videos = dataset.get_subset_videos('validation') + dataset.get_subset_videos('testing')

    print('Compiling model')
    model = RecurrentActivityClassificationNetwork(batch_size, timesteps, stateful=True, summary=True)
    model.load_weights(ACTIVITY_CLASSIFICATION['model_weights'])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print('Model Compiled!')

    nb_videos = len(videos)
    print('Total number of videos: {}\n'.format(nb_videos))
    count = 0
    for video in videos:
        count += 1
        print('{}/{}'.format(count, nb_videos))
        sys.stdout.flush()
        video_features_path = STORED_FEATURES_PATH + '/' + video.video_id + '.npy'
        features = np.load(video_features_path)
        nb_instances = features.shape[0]
        assert nb_instances == video.num_frames // 16
        features = features.reshape(nb_instances, 1, 4096)
        Y = model.predict(features, batch_size=batch_size)
        model.reset_states()
        Y = Y.reshape(nb_instances, output_shape)
        prediction_output_path = ACTIVITY_CLASSIFICATION['predictions_path'] + '/' + video.video_id + '.npy'
        np.save(prediction_output_path, Y)


if __name__ == '__main__':
    extract_predicted_outputs()
