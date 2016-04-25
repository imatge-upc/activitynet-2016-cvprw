import multiprocessing
import sys
import time

import numpy as np

from keras.preprocessing.image import list_pictures
from keras.utils.generic_utils import Progbar
from work.config import (STORED_FEATURES_PATH, STORED_VIDEOS_EXTENSION,
                         STORED_VIDEOS_PATH)
from work.dataset.activitynet import ActivityNetDataset
from work.models.c3d import C3D_conv_features
from work.training.generator import VideoGenerator


def extract_features():
    # Defining variables
    input_size = (112, 112)
    length = 16
    batch_size = 32
    max_q_size = 10
    nb_workers = 8
    wait_time = 0.1

    # Loading dataset
    print('Loading dataset')
    dataset = ActivityNetDataset(
        videos_path='../../dataset/videos.json',
        labels_path='../../dataset/labels.txt',
        stored_videos_path=STORED_VIDEOS_PATH,
        files_extension=STORED_VIDEOS_EXTENSION
    )
    # Removing the videos which already were extracted its features
    extracted_features_files = list_pictures(STORED_FEATURES_PATH, ext='npy')
    features_ids = [feature[:-4].split('/')[-1] for feature in extracted_features_files]
    print('Videos already downloaded: {} videos'.format(len(features_ids)))
    to_remove = []
    for video in dataset.videos:
        if video.video_id in features_ids:
            to_remove.append(video)
    for video in to_remove:
        dataset.videos.remove(video)
    nb_videos = len(dataset.videos)
    print('Total number of videos: {} videos'.format(nb_videos))

    # creating parallel data loading videos
    print('Creating {} process to fetch video data'.format(nb_workers))
    data_gen_queue = multiprocessing.Queue()
    _stop = multiprocessing.Event()
    def data_generator_task(index):
        generator = VideoGenerator(dataset.videos[index:nb_videos:nb_workers],
            STORED_VIDEOS_PATH, STORED_VIDEOS_EXTENSION, length, input_size)
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
                raise
    generator_threads = [multiprocessing.Process(target=data_generator_task, args=[i])
                         for i in range(nb_workers)]
    for process in generator_threads:
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
    mean_total = np.load('../../models/c3d/c3d-sports1M_mean.npy')
    mean = np.mean(mean_total, axis=(0, 2, 3, 4), keepdims=True)

    progbar = Progbar(nb_videos)
    counter = 0
    progbar.update(counter)
    while counter < nb_videos:
        counter += 1
        # print('Video {}/{}'.format(counter, len(dataset.videos)))
        # sys.stdout.flush()
        generator_output = None
        while not _stop.is_set():
            if not data_gen_queue.empty():
                generator_output = data_gen_queue.get()
                break
            else:
                time.sleep(wait_time)
        video_id, X = generator_output
        if X is None:
            print('Could not be read the video {}'.format(video_id))
            sys.stdout.flush()
            continue
        X = X - mean
        Y = model.predict(X, batch_size=batch_size)
        save_path = STORED_FEATURES_PATH + '/' + video_id + '.npy'
        np.save(save_path, Y)
        progbar.update(counter)
    _stop.set()

    print('Feature extraction completed')

if __name__ == '__main__':
    extract_features()
