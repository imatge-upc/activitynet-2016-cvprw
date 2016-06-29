import argparse
import multiprocessing
import os
import sys
import time

import h5py
import numpy as np
from progressbar import ProgressBar

from ..src.data import VideoGenerator


# from work.dataset.activitynet import ActivityNetDataset
# from work.environment import (DATASET_LABELS, DATASET_VIDEOS, STORED_FEATURES_PATH,
#                               STORED_MEAN_PATH, STORED_VIDEOS_EXTENSION, STORED_VIDEOS_PATH)
# from work.models.c3d import C3D_conv_features
# from work.processing.data import VideoGenerator


def extract_features(videos_dir, output_dir, batch_size, num_threads, queue_size, num_gpus):
    # Defining variables
    input_size = (112, 112)
    length = 16
    wait_time = 0.1


    output_file = h5py.File(os.path.join(output_dir, 'video_features.hdf5'), 'r+')
    videos_ids = [v[:-4] for v in os.listdir(videos_dir) if v[:-4] == '.mp4']

    # Lets remove from the list videos_ids, the ones already extracted its features
    videos_ids_to_extract = list(set(videos_ids) - set(output_file.keys()))

    nb_videos = len(videos_ids_to_extract)
    print('Total number of videos: {}'.format(len(videos_ids)))
    print('Videos already extracted its features: {}'.format(len(output_file.keys())))
    print('Videos to extract its features: {}'.format(len(nb_videos)))

    # Creating Parallel Fetching Video Data
    print('Creating {} process to fetch video data'.format(num_threads))
    data_gen_queue = multiprocessing.Queue(maxsize=queue_size)
    _stop = multiprocessing.Event()
    def data_generator_task(index):
        generator = VideoGenerator(videos_ids_to_extract[index:nb_videos:num_threads],
            videos_dir, '.mp4', length, input_size)
        while not _stop.is_set():
            try:
                if data_gen_queue.qsize() < max_q_size:
                    try:
                        generator_output = next(generator)
                    except ValueError:
                        continue
                    data_gen_queue.put(generator_output)
                else:
                    time.sleep(wait_time)
            except Exception:
                _stop.set()

    generator_process = [multiprocessing.Process(target=data_generator_task, args=[i])
                            for i in range(num_threads)]
    for process in generator_process:
        process.daemon = True
        process.start()



    # Loading the model
    print('Loading model')
    model = C3D_conv_features(summary=True)
    print('Compiling model')
    model.compile(optimizer='sgd', loss='mse')
    print('Compiling done!')

    print('Starting extracting features')
    print('Total number of videos to extract features: {} videos'.format(nb_videos))

    print('Loading mean')
    mean_total = np.load(STORED_MEAN_PATH)
    mean = np.mean(mean_total, axis=(0, 2, 3, 4), keepdims=True)

    progbar = ProgressBar(max_value=nb_videos)
    counter = 0
    progbar.update(counter)
    while counter < nb_videos:
        counter += 1
        # print('Video {}/{}'.format(counter, len(dataset.videos)))
        generator_output = None
        while not (_stop.is_set() and data_gen_queue.empty()):
            if not data_gen_queue.empty():
                generator_output = data_gen_queue.get()
                if not generator_output:
                    continue
                break
            else:
                time.sleep(wait_time)
        video_id, X = generator_output
        if X is None:
            print('Could not be read the video {}'.format(video_id))
            continue
        X = X - mean
        Y = model.predict(X, batch_size=batch_size)
        save_path = STORED_FEATURES_PATH + '/' + video_id + '.npy'
        np.save(save_path, Y)
        progbar.update(counter)
    _stop.set()
    progbar.finish()
    print('Feature extraction completed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract video features using C3D network')
    parser.add_argument('-d', '--videos-dir', type=str, destination='directory',
        help='videos directory')
    parser.add_argument('-o', '--output-dir', type=str, destination='output',
        help='directory where to store the extracted features')
    parser.add_argument('-b', '--batch-size', type=int, destination='batch_size',
        default=32, help='batch size when extracting features (default: %(default)s)')
    parser.add_argument('-t', '--num-threads', type=int, destination='num_threads',
        default=8, help='number of threads to fetch videos (default: %(default)s)')
    parser.add_argument('-q', '--queue-size', type=int, destination='queue_size',
        default=12, help='maximum number of elements at the queue when fetching videos (default %(default)s)')
    parser.add_argument('-g', '--num-gpus', type=int, destination='num_gpus',
        default=1, help='number of gpus to use for extracting features (default: %(default)s)')

    args = parser.parse_args()

    extract_features(args.directory, args.output, args.batch_size, args.num_threads, args.queue_size, args.num_gpus)
