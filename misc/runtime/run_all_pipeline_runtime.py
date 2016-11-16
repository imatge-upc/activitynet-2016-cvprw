import argparse
import os
import random
import time

import numpy as np
from keras.layers import (LSTM, BatchNormalization, Convolution3D, Dense, Dropout, Flatten, Input,
                          MaxPooling3D, TimeDistributed, ZeroPadding3D)
from keras.models import Model, Sequential

from src.data import import_labels
from src.io import get_duration, get_num_frames, video_to_array
from src.processing import activity_localization, get_classification, smoothing

runtime_measures = {
    'load_video': [],
    'extract_features_c3d': [],
    'temporal_localization_network': [],
    'post-processing': [],
    'video_duration': []
}


def run_runtime_tests(input_video, model_features, c3d_mean, model_localization):
    input_size = (112, 112)
    length = 16

    # Setup post-processing variables
    smoothing_k = 5
    activity_threshold = .2

    # Load labels
    with open('dataset/labels.txt', 'r') as f:
        labels = import_labels(f)

    print('')
    print('#'*50)
    print(input_video)
    print('Reading Video...')
    t_s = time.time()
    video_array = video_to_array(input_video, resize=input_size)
    t_e = time.time()
    print('Loading Video: {:.2f}s'.format(t_e-t_s))
    runtime_measures['load_video'].append(t_e-t_s)
    if video_array is None:
        raise Exception('The video could not be read')
    nb_frames = get_num_frames(input_video)
    duration = get_duration(input_video)
    fps = nb_frames / duration
    runtime_measures['video_duration'].append(duration)
    print('Duration: {:.1f}s'.format(duration))
    print('FPS: {:.1f}'.format(fps))
    print('Number of frames: {}'.format(nb_frames))

    nb_clips = nb_frames // length
    video_array = video_array.transpose(1, 0, 2, 3)
    video_array = video_array[:nb_clips*length,:,:,:]
    video_array = video_array.reshape((nb_clips, length, 3, 112, 112))
    video_array = video_array.transpose(0, 2, 1, 3, 4)

    # Extract features
    print('Extracting features...')
    t_s = time.time()
    X = video_array - c3d_mean
    Y = model_features.predict(X, batch_size=1, verbose=1)
    t_e = time.time()
    print('Extracting C3D features: {:.2f}s'.format(t_e-t_s))
    runtime_measures['extract_features_c3d'].append(t_e-t_s)

    # Predict with the temporal localization network
    print('Predicting...')
    t_s = time.time()
    Y = Y.reshape(nb_clips, 1, 4096)
    prediction = model_localization.predict(Y, batch_size=1, verbose=1)
    prediction = prediction.reshape(nb_clips, 201)
    t_e = time.time()
    print('Prediction temporal activities: {:.2f}s'.format(t_e-t_s))
    runtime_measures['temporal_localization_network'].append(t_e-t_s)

    # Post processing the predited output
    print('Post-processing output...')
    t_s = time.time()

    labels_idx, scores = get_classification(prediction, k=5)
    print('Video: {}\n'.format(input_video))
    print('Classification:')
    for idx, score in zip(labels_idx, scores):
        label = labels[idx]
        print('{:.4f}\t{}'.format(score, label))

    prediction_smoothed = smoothing(prediction, k=smoothing_k)
    activities_idx, startings, endings, scores = activity_localization(
        prediction_smoothed,
        activity_threshold
    )
    t_e = time.time()
    runtime_measures['post-processing'].append(t_e-t_s)
    print('Post-processing runtime: {:.2f}s'.format(t_e-t_s))

    print('\nDetection:')
    print('Score\tInterval\t\tActivity')
    for idx, s, e, score in zip(activities_idx, startings, endings, scores):
        start = s * float(length) / fps
        end = e * float(length) / fps
        label = labels[idx]
        print('{:.4f}\t{:.1f}s - {:.1f}s\t\t{}'.format(score, start, end, label))


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

def temporal_localization_network(summary=False):
    input_features = Input(batch_shape=(1, 1, 4096,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(p=.5)(input_normalized)
    lstm = LSTM(512, return_sequences=True, stateful=True, name='lsmt1')(input_dropout)
    output_dropout = Dropout(p=.5)(lstm)
    output = TimeDistributed(Dense(201, activation='softmax'), name='fc')(output_dropout)

    model = Model(input=input_features, output=output)
    model.load_weights('data/models/temporal-location_weights.hdf5')

    if summary:
        model.summary()
    return model


if __name__ == '__main__':
    videos_dir = '/imatge/amontes/work/datasets/ActivityNet/v1.3/videos'
    N = 20  # Number of random videos
    R = 3   # Repetitions

    # Read dataset and choose 10 random videos from test dataset
    videos_ids = [v for v in os.listdir(videos_dir) if v[-4:] == '.mp4']
    start_video = random.choice(videos_ids)
    videos_ids = random.sample(videos_ids, N)

    # Load C3D model and mean
    print('Loading C3D network...')
    model_features  = C3D_conv_features(True)
    model_features.compile(optimizer='sgd', loss='mse')
    mean_total = np.load('data/models/c3d-sports1M_mean.npy')
    c3d_mean = np.mean(mean_total, axis=(0, 2, 3, 4), keepdims=True)

    # Load the temporal localization network
    print('Loading temporal localization network...')
    model_localization = temporal_localization_network(True)
    model_localization.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # First lets start with a random video (The timing of the first video is not significant as
    # Keras working with Theano require to compile which increase the runnig time only the first time)
    video_path = os.path.join(videos_dir, start_video)
    run_runtime_tests(video_path, model_features, c3d_mean, model_localization)

    for i in range(N):
        video_path = os.path.join(videos_dir, videos_ids[i])
        for _ in range(R):
            run_runtime_tests(video_path, model_features, c3d_mean, model_localization)

    with open('runtime_2.out', 'w') as f:
        for k in runtime_measures.keys():
            f.write(k+';')
            values = runtime_measures[k]
            for v in values:
                f.write(str(v))
                f.write(';')
            f.write('\n')
