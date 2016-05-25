import h5py
from progressbar import ProgressBar

from keras.layers import LSTM, BatchNormalization, Dense, Input, TimeDistributed
from keras.models import Model
from work.environment import FEATURES_DATASET_FILE


def extract_predicted_outputs():

    print('Compiling model')
    input_features = Input(batch_shape=(1, 1, 4096,), name='features')
    input_normalized = BatchNormalization()(input_features)
    lstm1 = LSTM(1024, return_sequences=True, stateful=True, name='lstm1')(input_normalized)
    lstm2 = LSTM(1024, return_sequences=True, stateful=True, name='lstm2')(lstm1)
    lstm3 = LSTM(1024, return_sequences=True, stateful=True, name='lstm3')(lstm2)
    output = TimeDistributed(Dense(201, activation='softmax'), name='fc')(lstm3)
    model = Model(input=input_features, output=output)
    model.load_weights('model_snapshot/lstm_activity_classification_deeper_03_e030.hdf5')
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print('Model Compiled!')

    h5_dataset = h5py.File(FEATURES_DATASET_FILE, 'r')
    h5_predict = h5py.File('./predictions/predictions_deeper_e030.hdf5', 'w')

    total_nb = len(h5_dataset['validation'].keys()) + len(h5_dataset['testing'].keys())
    count = 0
    print('Predicting...')
    progbar = ProgressBar(max_value=total_nb)

    for subset in ('validation', 'testing'):
        subset_dataset = h5_dataset[subset]
        output_subset = h5_predict.create_group(subset)
        for video_id in subset_dataset.keys():
            progbar.update(count)
            video_features = subset_dataset[video_id][...]
            nb_instances = video_features.shape[0]
            video_features = video_features.reshape(nb_instances, 1, 4096)
            model.reset_states()
            Y = model.predict(video_features, batch_size=1)
            Y = Y.reshape(nb_instances, 201)

            output_subset.create_dataset(video_id, data=Y)
            count += 1

    progbar.finish()
    h5_dataset.close()
    h5_predict.close()

if __name__ == '__main__':
    extract_predicted_outputs()
