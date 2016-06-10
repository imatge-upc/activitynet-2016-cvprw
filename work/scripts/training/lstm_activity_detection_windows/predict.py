import h5py
from progressbar import ProgressBar

from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input
from keras.models import Model
from work.environment import FEATURES_DATASET_FILE


def extract_predicted_outputs():
    experiment = 1
    nb_epoch = 100
    subsets = ('validation',)
    video_features_size = 4096

    weights_path = 'model_snapshot/lstm_activity_detection_windows_{experiment:02d}_e{nb_epoch:03d}.hdf5'.format(
        experiment=experiment, nb_epoch=nb_epoch
    )
    store_file = './predictions/predictions_2_{experiment:02d}_e{nb_epoch:03d}.hdf5'.format(
        experiment=experiment, nb_epoch=nb_epoch
    )

    print('Compiling model')
    input_video_features = Input(batch_shape=(1, 1, video_features_size,),
        name='video_features')
    video_normalized = BatchNormalization(name='video_features_normalization')(input_video_features)

    input_dropout = Dropout(p=.5)(video_normalized)
    lstm = LSTM(512, return_sequences=False, stateful=True, name='lstm1')(input_dropout)

    detection = Dense(201, activation='sigmoid', name='activity_detection')(lstm)

    model = Model(input=input_video_features, output=detection)
    model.load_weights(weights_path)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    print('Model Compiled!')

    h5_dataset = h5py.File(FEATURES_DATASET_FILE, 'r')
    h5_predict = h5py.File(store_file, 'w')

    total_nb = 0
    for subset in subsets:
        total_nb += len(h5_dataset[subset].keys())
    count = 0
    print('Predicting...')
    progbar = ProgressBar(max_value=total_nb)

    for subset in subsets:
        subset_dataset = h5_dataset[subset]
        output_subset = h5_predict.create_group(subset)
        for video_id in subset_dataset.keys():
            progbar.update(count)
            video_features = subset_dataset[video_id][...]
            nb_instances = video_features.shape[0]
            video_features = video_features.reshape(nb_instances, 1, 4096)
            #model.reset_states()
            Y = model.predict(video_features, batch_size=1)
            Y = Y.reshape(nb_instances, 201)

            output_subset.create_dataset(video_id, data=Y)
            count += 1

    progbar.finish()
    h5_dataset.close()
    h5_predict.close()

if __name__ == '__main__':
    extract_predicted_outputs()
