import h5py
import numpy as np
from progressbar import ProgressBar

from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed, merge
from keras.models import Model
from work.environment import FEATURES_DATASET_FILE


def extract_predicted_outputs():
    experiment = 2
    nb_epoch = 100
    subsets = ('validation',)

    weights_path = 'model_snapshot/lstm_activity_classification_feedback_{experiment:02d}_e{nb_epoch:03d}.hdf5'.format(
        experiment=experiment, nb_epoch=nb_epoch
    )
    store_file = './predictions/predictions_{experiment:02d}_e{nb_epoch:03d}.hdf5'.format(
        experiment=experiment, nb_epoch=nb_epoch
    )

    print('Compiling model')
    input_features = Input(batch_shape=(1, 1, 4096,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    previous_output = Input(batch_shape=(1, 1, 202,), name='prev_output')
    merging = merge([input_normalized, previous_output], mode='concat', concat_axis=-1)
    merging_dropout = Dropout(p=0.5)(merging)
    lstm1 = LSTM(512, return_sequences=True, stateful=True, name='lstm1')(merging_dropout)
    lstm2 = LSTM(512, return_sequences=True, stateful=True, name='lstm2')(lstm1)
    output_dropout = Dropout(p=0.5)(lstm1)
    output = TimeDistributed(Dense(201, activation='softmax'), name='fc')(output_dropout)

    model = Model(input=[input_features, previous_output], output=output)
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print('Model Compiled!')

    h5_dataset = h5py.File(FEATURES_DATASET_FILE, 'r')
    h5_predict = h5py.File(store_file, 'w')

    total_nb = 0
    for subset in subsets:
        total_nb += len(h5_dataset[subset].keys())
    count = 0
    print('Predicting...')
    progbar = ProgressBar(max_value=total_nb)

    for subset in ('validation',):
        subset_dataset = h5_dataset[subset]
        output_subset = h5_predict.create_group(subset)
        for video_id in subset_dataset.keys():
            progbar.update(count)
            video_features = subset_dataset[video_id][...]
            nb_instances = video_features.shape[0]
            video_features = video_features.reshape(nb_instances, 1, 4096)
            Y = np.zeros((nb_instances, 201))
            X_prev_output = np.zeros((1, 202))
            X_prev_output[0,201] = 1
            model.reset_states()
            for i in range(nb_instances):
                X_features = video_features[i,:,:].reshape(1, 1, 4096)
                X_prev_output = X_prev_output.reshape(1, 1, 202)
                next_output = model.predict_on_batch(
                    {'features': X_features,
                    'prev_output': X_prev_output}
                )
                Y[i,:] = next_output[0,:]
                X_prev_output = np.zeros((1, 202))
                X_prev_output[0,:201] = next_output[0,:]

            output_subset.create_dataset(video_id, data=Y)
            count += 1

    progbar.finish()
    h5_dataset.close()
    h5_predict.close()

if __name__ == '__main__':
    extract_predicted_outputs()
