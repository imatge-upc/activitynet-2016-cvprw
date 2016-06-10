import h5py
import numpy as np
from progressbar import ProgressBar

from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed, merge
from keras.models import Model
from work.environment import FEATURES_DATASET_FILE


def extract_predicted_outputs():
    experiment = 4
    nb_epoch = 150
    subsets = ('validation',)
    vid_size = 4096
    mfcc_size = 80
    spec_size = 8

    weights_path = 'model_snapshot/lstm_activity_classification_{experiment:02d}_e{nb_epoch:03d}.hdf5'.format(
        experiment=experiment, nb_epoch=nb_epoch
    )
    store_file = './predictions/predictions_{experiment:02d}_e{nb_epoch:03d}.hdf5'.format(
        experiment=experiment, nb_epoch=nb_epoch
    )

    print('Compiling model')
    input_video_features = Input(batch_shape=(1, 1, vid_size,),
        name='video_features')
    input_video_normalized = BatchNormalization(name='video_normalization')(input_video_features)
    input_mfcc_features = Input(batch_shape=(1, 1, mfcc_size,),
        name='mfcc_features')
    input_mfcc_normalized = BatchNormalization(name='mfcc_normalization')(input_mfcc_features)
    input_spec_features = Input(batch_shape=(1, 1, spec_size,),
        name='spec_features')
    input_spec_normalized = BatchNormalization(name='spec_normalization')(input_spec_features)
    input_merged = merge([input_video_normalized, input_mfcc_normalized, input_spec_normalized],
        mode='concat', concat_axis=-1)
    input_dropout = Dropout(p=.5)(input_merged)
    lstm = LSTM(512, return_sequences=True, stateful=True, name='lstm1')(input_dropout)
    #lstm_output = LSTM(512, return_sequences=True, stateful=True, name='lstm2')(lstm)
    lstm_dropout = Dropout(p=.5)(lstm)
    output = TimeDistributed(Dense(201, activation='softmax'), name='softmax')(lstm_dropout)

    model = Model(input=[input_video_features, input_mfcc_features, input_spec_features],
        output=output)
    model.load_weights(weights_path)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print('Model Compiled!')

    h5_dataset = h5py.File(FEATURES_DATASET_FILE, 'r')
    h5_dataset_audio = h5py.File('../dataset/audio_descriptors.hdf5', 'r')
    h5_predict = h5py.File(store_file, 'w')

    videos_ids = []

    for subset in subsets:
        videos = h5_dataset[subset].keys()
        videos = set(videos).intersection(h5_dataset_audio['mfcc'].keys())
        videos = videos.intersection(h5_dataset_audio['spec'].keys())
        videos = list(videos)
        videos_ids.append(videos)

    total_nb = 0
    for x in videos_ids:
        total_nb += len(x)

    count = 0
    print('Predicting...')
    progbar = ProgressBar(max_value=total_nb)

    for subset, videos_ids in zip(subsets, videos_ids):
        output_subset = h5_predict.create_group(subset)
        for video_id in videos_ids:
            progbar.update(count)
            vid_features = h5_dataset[subset][video_id][...]
            nb_instances = vid_features.shape[0]
            mfcc_features = h5_dataset_audio['mfcc'][video_id][...]
            spec_features = h5_dataset_audio['spec'][video_id][...]
            spec_features = np.broadcast_to(spec_features, (nb_instances, spec_size))

            vid_features = vid_features.reshape(nb_instances, 1, vid_size)
            mfcc_features = mfcc_features.reshape(nb_instances, 1, mfcc_size)
            spec_features = spec_features.reshape(nb_instances, 1, spec_size)
            model.reset_states()
            Y = model.predict({'video_features': vid_features,
                                'mfcc_features': mfcc_features,
                                'spec_features': spec_features}, batch_size=1)
            Y = Y.reshape(nb_instances, 201)

            output_subset.create_dataset(video_id, data=Y)
            count += 1

    progbar.finish()
    h5_dataset.close()
    h5_dataset_audio.close()
    h5_predict.close()

if __name__ == '__main__':
    extract_predicted_outputs()
