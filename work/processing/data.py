import random
import sys
import threading
import time

import numpy as np
from progressbar import ProgressBar

from keras.preprocessing.image import (horizontal_flip, random_rotation,
                                       random_shear, random_shift,
                                       vertical_flip)
from keras.preprocessing.video import temporal_flip, video_to_array
from keras.utils import np_utils
from work.environment import STORED_FEATURES_PATH


class VideoGenerator(object):

    def __init__(self, videos, stored_videos_path,
            stored_videos_extension, length, input_size):
        self.videos = videos
        self.total_nb_videos = len(videos)
        self.flow_generator = self._flow_index(self.total_nb_videos)
        self.lock = threading.Lock()
        self.stored_videos_path = stored_videos_path
        self.stored_videos_extension = stored_videos_extension
        self.length = length
        self.input_size = input_size

    def _flow_index(self, total_nb_videos):
        pointer = 0
        while pointer < total_nb_videos:
            pointer += 1
            yield pointer-1

    def next(self):
        with self.lock:
            index = next(self.flow_generator)
        t1 = time.time()
        video = self.videos[index]
        nb_instances = video.num_frames // self.length
        path = self.stored_videos_path + '/' + video.video_id + '.' + self.stored_videos_extension
        vid_array = video_to_array(path, start_frame=0,
                                   end_frame=self.length*nb_instances,
                                   resize=self.input_size)
        if vid_array is not None:
            vid_array = vid_array.transpose(1, 0, 2, 3)
            vid_array = vid_array.reshape((nb_instances, self.length, 3,)+(self.input_size))
            vid_array = vid_array.transpose(0, 2, 1, 3, 4)
        t2 = time.time()
        print('Time to fetch {} video: {:.2f} seconds'.format(video.video_id, t2-t1))
        sys.stdout.flush()
        return video.video_id, vid_array

    def __next__(self):
        self.next()

class VideoDatasetGenerator(object):


    def __init__(self,
                 samplewise_center=True,
                 samplewise_std_normalization=True,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 temporal_flip=False):
        self.__dict__.update(locals())
        self.mean = None
        self.std = None
        self.principal_components = None
        self.lock = threading.Lock()


    def random_transform(self, x):
        if self.rotation_range:
            x = random_rotation(x, self.rotation_range)
        if self.width_shift_range or self.height_shift_range:
            x = random_shift(x, self.width_shift_range, self.height_shift_range)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = horizontal_flip(x)
        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = vertical_flip(x)
        if self.temporal_flip:
            if np.random.random() < 0.5:
                x = temporal_flip(x)
        if self.shear_range:
            x = random_shear(x, self.shear_range)
        return x

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see # http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.flow_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        bX = np.zeros((current_batch_size, self.channels, self.length, self.size[0], self.size[1]))
        y = []
        t1 = time.time()
        for i, j in enumerate(index_array):
            instance = self.instances[j]
            path = self.save_path + '/{}.{}'.format(instance.instance_id, self.files_extension)
            x = video_to_array(path, resize=self.size, start_frame=instance.start_frame, length=self.length)
            x = self.random_transform(x.astype('float32'))
            x = self.standardize(x)
            bX[i] = x
            y.append(instance.output)

        bY = np_utils.to_categorical(y, nb_classes=self.nb_classes)
        t2 = time.time()
        print('Time to fetch a batch: {:.2f} seconds'.format(t2-t1))
        sys.stdout.flush()
        return bX, bY

    def __next__(self):
        self.next()

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        b = 0
        total_b = 0
        while 1:
            if b == 0:
                if seed is not None:
                    np.random.seed(seed + total_b)

                if shuffle:
                    index_array = np.random.permutation(N)
                else:
                    index_array = np.arange(N)

            current_index = (b * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
            else:
                current_batch_size = N - current_index

            if current_batch_size == batch_size:
                b += 1
            else:
                b = 0
            total_b += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def flow(self, dataset, subset, size=(128, 171), length=16, channels=3, batch_size=32, shuffle=False, seed=None):
        self.instances = getattr(dataset, 'instances_{}'.format(subset))
        self.save_path = dataset.stored_videos_path
        self.files_extension = dataset.files_extension

        self.size = size
        self.length = length
        self.channels = channels
        self.nb_classes = dataset.num_classes
        self.flow_generator = self._flow_index(len(self.instances),
            batch_size, shuffle, seed)
        return self

    def standardize(self, x):
        if self.samplewise_center:
            x -= np.mean(x, axis=1, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=1, keepdims=True) + 1e-7)
        return x


def load_features_data(videos, timesteps, batch_size, output_mode='all'):
    """ The output data can be loaded in three modes:
    * all: the output dimension is 201 which include all the activities and the non activity.
    * activity: the output dimension is 200 which represent the 200 activities of the dataset. ** Not implemented **
    * binary_activity: the output dimension is 2 which represent wether or not there is an activity present
    """
    if output_mode not in ('all', 'binary_activity'):
        raise Exception('output values not valid')

    length = 16
    features_size = 4096
    if output_mode == 'all':
        output_size = 201
    elif output_mode == 'binary_activity':
        output_size = 2

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
        output_classes = []
        for instance in video.instances:
            if output_mode == 'all':
                output_classes.append(instance.output)
            elif output_mode == 'binary_activity':
                output_classes.append(instance.activity_binary_output)
        output[pos:pos+nb_video_instances,:] = np_utils.to_categorical(output_classes, nb_classes=output_size)
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
