from __future__ import absolute_import

import argparse
import multiprocessing
import os
import sys
import time
import traceback

import h5py
import numpy as np
from progressbar import ProgressBar

from src.data import VideoGenerator


def extract_features(videos_dir, output_dir, batch_size, num_threads, queue_size, num_gpus):
    # Defining variables
    input_size = (112, 112)
    length = 16
    wait_time = 0.1

    output_path = os.path.join(output_dir, 'video_features.hdf5')
    mode = 'r+' if os.path.exists(output_path) else 'w'
    # Extract the ids of the videos already extracted its features
    output_file = h5py.File(output_path, mode)
    extracted_videos = output_file.keys()
    output_file.close()

    videos_ids = [v[:-4] for v in os.listdir(videos_dir) if v[-4:] == '.mp4']

    # Lets remove from the list videos_ids, the ones already extracted its features
    videos_ids_to_extract = list(set(videos_ids) - set(extracted_videos))

    nb_videos = len(videos_ids_to_extract)
    print('Total number of videos: {}'.format(len(videos_ids)))
    print('Videos already extracted its features: {}'.format(len(extracted_videos)))
    print('Videos to extract its features: {}'.format(nb_videos))

    # Creating Parallel Fetching Video Data
    print('Creating {} process to fetch video data'.format(num_threads))
    data_gen_queue = multiprocessing.Queue(maxsize=queue_size)
    _stop_all_generators = multiprocessing.Event()
    _stop_all_extractors = multiprocessing.Event()
    def data_generator_task(index):
        generator = VideoGenerator(videos_ids_to_extract[index:nb_videos:num_threads],
            videos_dir, 'mp4', length, input_size)
        keep = True
        while keep:
            try:
                if data_gen_queue.qsize() < queue_size:
                    try:
                        generator_output = next(generator)
                    except ValueError:
                        continue
                    data_gen_queue.put(generator_output)
                else:
                    time.sleep(wait_time)
            except StopIteration:
                print('End')
                break
            except Exception:
                keep = False
                print('Something went wrong with generator_process')
                print(traceback.print_exc())

    generator_process = [multiprocessing.Process(target=data_generator_task, args=[i])
                            for i in range(num_threads)]
    for process in generator_process:
        process.daemon = True
        process.start()


    data_save_queue = multiprocessing.Queue()
    def extranting_features_task():
        # Loading the model
        print('Loading model')
        model = C3D_conv_features(summary=True)
        print('Compiling model')
        model.compile(optimizer='sgd', loss='mse')
        print('Compiling done!')

        print('Starting extracting features')

        print('Loading mean')
        mean_total = np.load('data/models/c3d-sports1M_mean.npy')
        mean = np.mean(mean_total, axis=(0, 2, 3, 4), keepdims=True)

        while not (_stop_all_generators.is_set() and data_gen_queue.empty()):
            generator_output = None
            while True:
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
            data_save_queue.put((video_id, Y))
            print('Extracted features from video {}'.format(video_id))

    extractors_process = [multiprocessing.Process(target=extranting_features_task)
        for i in range(num_gpus)]
    for p in extractors_process:
        p.daemon = True
        p.start()

    # Create the process that will get all the extracted features from the data_save_queue and
    # store it on the hdf5 file.

    def saver_task():
        while not (_stop_all_extractors.is_set() and data_save_queue.empty()):
            extracted_output = None
            while True:
                if not data_save_queue.empty():
                    extracted_output = data_save_queue.get()
                    if not extracted_output:
                        continue
                    break
                else:
                    time.sleep(wait_time)
            video_id, features = extracted_output
            if features is None:
                print('Something went wrong')
                continue
            assert features.shape[1] == 4096
            with h5py.File(output_path, 'r+') as f:
                f.create_dataset(video_id, data=features, dtype='float32')
            print('Saved video {}'.format(video_id))

    saver_process = multiprocessing.Process(target=saver_task)
    saver_process.daemon = True
    saver_process.start()

    # Joining processes
    for p in generator_process:
        p.join()
    _stop_all_generators.set()
    for p in extractors_process:
        p.join()
    _stop_all_extractors.set()
    saver_process.join()


def C3D_conv_features(summary=False):
    """ Return the Keras model of the network until the fc6 layer where the
    convolutional features can be extracted.
    """
    from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
    from keras.layers.core import Dense, Dropout, Flatten
    from keras.models import Sequential

    model = Sequential()
    # 1st layer group
    model.add(Convolution3D(64, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv1',
                            subsample=(1, 1, 1),
                            input_shape=(3, 16, 112, 112),
                            trainable=False))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv2',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3a',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3b',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4a',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4b',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5a',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5b',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropadding'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool5'))
    model.add(Flatten(name='flatten'))
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6', trainable=False))
    model.add(Dropout(.5, name='do1'))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5, name='do2'))
    model.add(Dense(487, activation='softmax', name='fc8'))

    # Load weights
    model.load_weights('data/models/c3d-sports1M_weights.h5')

    for _ in range(4):
        model.pop_layer()

    if summary:
        print(model.summary())
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract video features using C3D network')
    parser.add_argument('-d', '--videos-dir', type=str, dest='directory',
        default='data/videos', help='videos directory (default: %(default)s)')
    parser.add_argument('-o', '--output-dir', type=str, dest='output',
        default='data/dataset', help='directory where to store the extracted features (default: %(default)s)')
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size',
        default=32, help='batch size when extracting features (default: %(default)s)')
    parser.add_argument('-t', '--num-threads', type=int, dest='num_threads',
        default=8, help='number of threads to fetch videos (default: %(default)s)')
    parser.add_argument('-q', '--queue-size', type=int, dest='queue_size',
        default=12, help='maximum number of elements at the queue when fetching videos (default %(default)s)')
    parser.add_argument('-g', '--num-gpus', type=int, dest='num_gpus',
        default=1, help='number of gpus to use for extracting features (default: %(default)s)')

    args = parser.parse_args()

    extract_features(
        args.directory,
        args.output,
        args.batch_size,
        args.num_threads,
        args.queue_size,
        args.num_gpus
    )
