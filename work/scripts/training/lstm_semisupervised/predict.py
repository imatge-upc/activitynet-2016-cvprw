import h5py
import numpy as np
from progressbar import ProgressBar

from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed, merge
from keras.models import Model
from work.environment import FEATURES_DATASET_FILE


def extract_predicted_outputs():
    experiment = 1
    nb_epoch = 70
    subsets = ('validation',)
    vid_size = 4096
    mfcc_size = 80
    spec_size = 8

    weights_path = 'model_snapshot/lstm_semisupervised_{experiment:02d}_e{nb_epoch:03d}.hdf5'.format(
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
    input_previous_output = Input(batch_shape=(1, 1, 202,), name='prev_output')
    input_merged = merge([input_video_normalized, input_mfcc_normalized, input_spec_normalized,
        input_previous_output], mode='concat', concat_axis=-1)

    input_dropout = Dropout(p=0.5)(input_merged)
    lstm1 = LSTM(512, return_sequences=True, stateful=True, name='lstm1')(input_dropout)
    # lstm2 = LSTM(512, return_sequences=True, stateful=True, name='lstm2')(lstm1)
    output_dropout = Dropout(p=0.5)(lstm1)
    output = TimeDistributed(Dense(201, activation='softmax'), name='fc')(output_dropout)
    model = Model(input=[input_video_features, input_mfcc_features, input_spec_features, input_previous_output], output=output)

    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print('Model Compiled!')

    h5_dataset = h5py.File(FEATURES_DATASET_FILE, 'r')
    h5_dataset_audio = h5py.File('../dataset/audio_descriptors.hdf5', 'r')
    h5_predict = h5py.File(store_file, 'w')

    videos = []
    for subset in subsets:
        videos_ids = h5_dataset[subset].keys()
        videos_ids = set(videos_ids).intersection(h5_dataset_audio['mfcc'].keys())
        videos_ids = videos_ids.intersection(h5_dataset_audio['spec'].keys())
        videos_ids = list(videos_ids)
        videos.append(videos_ids)

    total_nb = 0
    for x in videos:
        total_nb += len(x)

    count = 0
    print('Predicting...')
    progbar = ProgressBar(max_value=total_nb)

    for subset, videos_ids in zip(subsets, videos):
        output_subset = h5_predict.create_group(subset)
        for video_id in videos_ids:
            progbar.update(count)
            video_features = h5_dataset[subset][video_id][...]
            nb_instances = video_features.shape[0]
            mfcc_features = h5_dataset_audio['mfcc'][video_id][...]
            spec_features = h5_dataset_audio['spec'][video_id][...]
            spec_features = np.broadcast_to(spec_features, (nb_instances, spec_size))

            video_features = video_features.reshape(nb_instances, 1, vid_size)
            mfcc_features = mfcc_features.reshape(nb_instances, 1, mfcc_size)
            spec_features = spec_features.reshape(nb_instances, 1, spec_size)
            Y = np.zeros((nb_instances, 201))
            X_prev_output = np.zeros((1, 202))
            X_prev_output[0,201] = 1
            model.reset_states()
            for i in range(nb_instances):
                X_features = video_features[i,:,:].reshape(1, 1, 4096)
                X_mfcc_features = mfcc_features[i,:,:].reshape(1, 1, mfcc_size)
                X_spec_features = spec_features[i,:,:].reshape(1, 1, spec_size)
                X_prev_output = X_prev_output.reshape(1, 1, 202)
                next_output = model.predict_on_batch(
                    {'video_features': X_features,
                    'mfcc_features': X_mfcc_features,
                    'spec_features': X_spec_features,
                    'prev_output': X_prev_output}
                )
                Y[i,:] = next_output[0,:]
                X_prev_output = np.zeros((1, 202))
                X_prev_output[0,:201] = next_output[0,:]

            output_subset.create_dataset(video_id, data=Y)
            count += 1

    progbar.finish()
    h5_dataset.close()
    h5_dataset_audio.close()
    h5_predict.close()

if __name__ == '__main__':
    extract_predicted_outputs()
