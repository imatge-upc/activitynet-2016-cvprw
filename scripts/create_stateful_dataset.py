import argparse
import json
import os
import random

import numpy as np
from progressbar import ProgressBar

import h5py
from src.data import generate_output, import_labels, to_categorical


def create_stateful_dataset(video_features_file, videos_info, labels,
                            output_path, batch_size, timesteps,
                            subset=None):
    features_size = 4096
    output_size = 201

    f_video_features = h5py.File(video_features_file, 'r')
    output_file = os.path.join(output_path, 'dataset_stateful.hdf5')
    f_dataset = h5py.File(output_file, 'w')

    if not subset:
        subsets = ['training', 'validation']
    else:
        subsets = [subset]


    with open(labels, 'r') as f:
        labels = import_labels(f)

    with open(videos_info, 'r') as f:
        videos_data = json.load(f)

    for subset in subsets:
        videos = [k for k in videos_data.keys() if videos_data[k]['subset'] == subset]
        videos = list(set(videos) & set(f_video_features.keys()))
        random.shuffle(videos)

        nb_videos = len(videos)
        print('Number of videos for {} subset: {}'.format(subset, nb_videos))

        # Check how the videos are going to be placed
        sequence_stack = []
        for _ in range(batch_size):
            sequence_stack.append([])
        nb_clips_stack = np.zeros(batch_size).astype(np.int64)
        accumulative_clips_stack = []
        for _ in range(batch_size):
            accumulative_clips_stack.append([])

        for video_id in videos:
            min_pos = np.argmin(nb_clips_stack)
            sequence_stack[min_pos].append(video_id)
            nb_clips_stack[min_pos] += f_video_features[video_id].shape[0]
            accumulative_clips_stack[min_pos].append(nb_clips_stack[min_pos])


        min_sequence = np.min(nb_clips_stack)
        max_sequence = np.max(nb_clips_stack)
        nb_batches_long = max_sequence // timesteps + 1
        nb_batches = min_sequence // timesteps
        print('Number of batches: {}'.format(nb_batches))

        video_features = np.zeros((nb_batches_long*batch_size*timesteps, features_size))
        output = np.zeros((nb_batches_long*batch_size*timesteps, output_size))
        index = np.arange(nb_batches_long*batch_size*timesteps)

        progbar = ProgressBar(max_value=batch_size)
        print('Creating stateful dataset for {} subset'.format(subset))

        for i in range(batch_size):
            batch_index = index // timesteps % batch_size == i
            progbar.update(i)

            pos = 0
            for video_id in sequence_stack[i]:
                # Video features
                vid_features = f_video_features[video_id][...]
                assert vid_features.shape[1] == features_size
                nb_instances = vid_features.shape[0]


                # Output
                output_classes = generate_output(videos_data[video_id], labels)
                assert nb_instances == len(output_classes)


                video_index = index[batch_index][pos:pos+nb_instances]
                video_features[video_index,:] = vid_features
                output[video_index] = to_categorical(output_classes, nb_classes=output_size)

                pos += nb_instances

        progbar.finish()

        video_features = video_features[:nb_batches*batch_size*timesteps,:]
        assert np.all(np.any(video_features, axis=1))
        video_features = video_features.reshape((nb_batches*batch_size, timesteps, features_size))

        output = output[:nb_batches*batch_size*timesteps,:]
        assert np.all(np.any(output, axis=1))
        output = output.reshape((nb_batches*batch_size, timesteps, output_size))

        if subset == 'training':
            background_weight = 0.6
            sample_weights = np.ones(output.shape[:2])
            sample_weights[output[:,:,0] == 1] = background_weight
        f_dataset_subset = f_dataset.create_group(subset)

        f_dataset_subset.create_dataset('vid_features', data=video_features, chunks=(4, timesteps, features_size), dtype='float32')
        f_dataset_subset.create_dataset('output', data=output, chunks=(batch_size, timesteps, output_size), dtype='float32')
        if subset == 'training':
            f_dataset_subset.create_dataset('sample_weight', data=sample_weights, chunks=(batch_size, timesteps), dtype='float32')

    f_dataset.close()
    f_video_features.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Put all the videos features into the correct way to train a RNN in a stateful way')
    parser.add_argument('-i', '--video-features', type=str, dest='video_features_file', default='data/dataset/video_features.hdf5', help='HDF5 where the video features have been extracted (default: %(default)s)')
    parser.add_argument('-v', '--videos-info', type=str, dest='videos_info', default='dataset/videos.json', help='File containing the annotations of all the videos on the dataset (default: %(default)s)')
    parser.add_argument('-l', '--labels', type=str, dest='labels', default='dataset/labels.txt', help='File containing the labels of the whole dataset (default: %(default)s)')
    parser.add_argument('-o', '--output-dir', type=str, dest='output_dir',
        default='data/dataset', help='directory where to store the stateful dataset (default: %(default)s)')
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=256, help='batch size desired to use for training (default: %(default)s)')
    parser.add_argument('-t', '--timesteps', type=int, dest='timesteps', default=20, help='timesteps desired for training the RNN (default: %(default)s)')
    parser.add_argument('-s', '--subset', type=str, dest='subset', default=None, choices=['training', 'validation'], help='Subset you want to create the stateful dataset (default: training and validation)')

    args = parser.parse_args()

    create_stateful_dataset(
        args.video_features_file,
        args.videos_info,
        args.labels,
        args.output_dir,
        args.batch_size,
        args.timesteps,
        args.subset
    )
