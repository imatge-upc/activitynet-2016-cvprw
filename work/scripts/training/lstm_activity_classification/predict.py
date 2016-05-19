import h5py
from progressbar import ProgressBar

from work.environment import FEATURES_DATASET_FILE
from work.models.decoder import RecurrentActivityClassificationNetwork


def extract_predicted_outputs():

    print('Compiling model')
    model = RecurrentActivityClassificationNetwork(1, 1, stateful=True, summary=True)
    model.load_weights('model_snapshot/lstm_activity_classification_02_e100.hdf5')
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print('Model Compiled!')

    h5_dataset = h5py.File(FEATURES_DATASET_FILE, 'r')
    h5_predict = h5py.File('./predictions/predictions_e100.hdf5', 'w')

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
