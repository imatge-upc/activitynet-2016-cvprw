import threading
import time

import numpy as np

from src.io import video_to_array


def import_labels(f):
    ''' Read from a file all the labels from it '''
    lines = f.readlines()
    labels = []
    i = 0
    for l in lines:
        t = l.split('\t')
        assert int(t[0]) == i
        label = t[1].split('\n')[0]
        labels.append(label)
        i += 1
    return labels

def to_categorical(y, nb_classes=None):
    ''' Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

def generate_output(video_info, labels, length=16):
    ''' Given the info of the vide, generate a vector of classes corresponding
    the output for each clip of the video which features have been extracted.
    '''
    nb_frames = video_info['num_frames']
    last_first_name = nb_frames - length + 1

    start_frames = range(0, last_first_name, length)

    # Check the output for each frame of the video
    outputs = ['none'] * nb_frames
    for i in range(nb_frames):
        # Pass frame to temporal scale
        t = i / float(nb_frames) * video_info['duration']
        for annotation in video_info['annotations']:
            if t >= annotation['segment'][0] and t <= annotation['segment'][1]:
                outputs[i] = annotation['label']
                label = annotation['label']
                break

    instances = []
    for start_frame in start_frames:
        # Obtain the label for this isntance and then its output
        output = None

        outs = outputs[start_frame:start_frame+length]
        if outs.count(label) >= length / 2:
            output = labels.index(label)
        else:
            output = 0
        instances.append(output)

    return instances


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
        video_id = self.videos[index]
        path = self.stored_videos_path + '/' + video_id + '.' + self.stored_videos_extension
        vid_array = video_to_array(path, start_frame=0,
                                   resize=self.input_size)
        if vid_array is not None:
            vid_array = vid_array.transpose(1, 0, 2, 3)
            nb_frames = vid_array.shape[0]
            nb_instances = nb_frames // self.length
            vid_array = vid_array[:nb_instances*self.length,:,:,:]
            vid_array = vid_array.reshape((nb_instances, self.length, 3,)+(self.input_size))
            vid_array = vid_array.transpose(0, 2, 1, 3, 4)
        t2 = time.time()
        print('Time to fetch {} video: {:.2f} seconds'.format(video_id, t2-t1))
        return video_id, vid_array

    def __next__(self):
        self.next()
