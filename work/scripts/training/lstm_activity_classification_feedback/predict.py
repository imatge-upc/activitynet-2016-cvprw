import h5py
import numpy as np
from progressbar import ProgressBar

from work.environment import FEATURES_DATASET_FILE
from work.models.decoder import RecurrentFeedbackActivityDetectionNetwork


def extract_predicted_outputs():

    print('Compiling model')
    model = RecurrentFeedbackActivityDetectionNetwork(1, 1, stateful=True, summary=True)
    model.load_weights('model_snapshot/lstm_activity_classification_feedback_02_e100.hdf5')
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print('Model Compiled!')

    h5_dataset = h5py.File(FEATURES_DATASET_FILE, 'r')
    h5_predict = h5py.File('./predictions/predictions_e100.hdf5', 'w')

    total_nb = len(h5_dataset['validation'].keys()) + len(h5_dataset['testing'].keys())
    count = 0
    progbar = ProgressBar(max_value=total_nb)

    print('Predicting...')
    for subset in ('validation', 'testing'):
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
