import sys

import numpy as np

from work.config import (RECURRENT_MODEL_WEIGHTS, STORED_FEATURES_PATH,
                         STORED_PREDICTION_PATH)
from work.dataset.activitynet import ActivityNetDataset
from work.models.decoder import RecurrentNetwork


def extract_predicted_outputs():
    batch_size = 1
    timesteps = 1

    print('Loading dataset')
    dataset = ActivityNetDataset(
        videos_path='../../dataset/videos.json',
        labels_path='../../dataset/labels.txt'
    )
    videos = dataset.get_subset_videos('validation')

    print('Compiling model')
    model = RecurrentNetwork(batch_size, timesteps, stateful=True, summary=True)
    model.load_weights(RECURRENT_MODEL_WEIGHTS)
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
        Y = model.predict_classes(features, batch_size=batch_size)
        model.reset_states()
        Y = Y.reshape(nb_instances)
        prediction_output_path = STORED_PREDICTION_PATH + '/' + video.video_id + '.npy'
        np.save(prediction_output_path, Y)


if __name__ == '__main__':
    extract_predicted_outputs()
