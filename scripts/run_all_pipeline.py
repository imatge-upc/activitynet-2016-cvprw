import argparse

import numpy as np

from keras.layers import (LSTM, BatchNormalization, Convolution3D, Dense, Dropout, Flatten, Input,
                          MaxPooling3D, TimeDistributed, ZeroPadding3D)
from keras.models import Model, Sequential
from src.data import import_labels
from src.io import get_duration, get_num_frames, video_to_array
from src.processing import activity_localization, get_classification, smoothing


def run_all_pipeline(input_video, smoothing_k, activity_threshold):
    input_size = (112, 112)
    length = 16

    # Load labels
    with open('dataset/labels.txt', 'r') as f:
        labels = import_labels(f)

    print('Reading Video...')
    video_array = video_to_array(input_video, resize=input_size)
    if video_array is None:
        raise Exception('The video could not be read')
    nb_frames = get_num_frames(input_video)
    duration = get_duration(input_video)
    fps = nb_frames / duration
    print('Duration: {:.1f}s'.format(duration))
    print('FPS: {:.1f}'.format(fps))
    print('Number of frames: {}'.format(nb_frames))

    nb_clips = nb_frames // length
    video_array = video_array.transpose(1, 0, 2, 3)
    video_array = video_array[:nb_clips*length,:,:,:]
    video_array = video_array.reshape((nb_clips, length, 3, 112, 112))
    video_array = video_array.transpose(0, 2, 1, 3, 4)

    # Load C3D model and mean
    print('Loading C3D network...')
    model  = C3D_conv_features(True)
    model.compile(optimizer='sgd', loss='mse')
    mean_total = np.load('data/models/c3d-sports1M_mean.npy')
    mean = np.mean(mean_total, axis=(0, 2, 3, 4), keepdims=True)

    # Extract features
    print('Extracting features...')
    X = video_array - mean
    Y = model.predict(X, batch_size=1, verbose=1)

    # Load the temporal localization network
    print('Loading temporal localization network...')
    model_localization = temporal_localization_network(True)
    model_localization.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # Predict with the temporal localization network
    print('Predicting...')
    Y = Y.reshape(nb_clips, 1, 4096)
    prediction = model_localization.predict(Y, batch_size=1, verbose=1)
    prediction = prediction.reshape(nb_clips, 201)

    # Post processing the predited output
    print('Post-processing output...')
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
    parser = argparse.ArgumentParser(description='Run all pipeline. Given a video, classify it and temporal localize the activity on it')

    parser.add_argument('-i', '--input-video', type=str, dest='input_video', help='Path to the input video')
    parser.add_argument('-k', type=int, dest='smoothing_k', default=5, help='Smoothing factor at post-processing (default: %(default)s)')
    parser.add_argument('-t', type=float, dest='activity_threshold', default=.2, help='Activity threshold at post-processing (default: %(default)s)')

    args = parser.parse_args()

    run_all_pipeline(
        args.input_video,
        args.smoothing_k,
        args.activity_threshold
    )
