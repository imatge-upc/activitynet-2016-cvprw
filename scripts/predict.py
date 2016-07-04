import argparse
import json
import os

import h5py
from progressbar import ProgressBar

from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed
from keras.models import Model


def extract_predicted_outputs(experiment_id, input_dataset, num_cells, num_layers, epoch, output_path, subset=None):

    if subset == None:
        subsets = ['validation', 'testing']
    else:
        subsets = [subset]

    weights_path = 'data/model_snapshot/lstm_activity_classification_{experiment_id}_e{nb_epoch:03d}.hdf5'.format(
        experiment_id=experiment_id, nb_epoch=epoch
    )
    store_file = 'predictions_{experiment_id}.hdf5'.format(
        experiment_id=experiment_id, nb_epoch=epoch
    )
    store_path = os.path.join(output_path, store_file)

    print('Compiling model')
    input_features = Input(batch_shape=(1, 1, 4096,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(p=0.5)(input_normalized)
    lstms_inputs = [input_dropout]
    for i in range(num_layers):
        previous_layer = lstms_inputs[-1]
        lstm = LSTM(num_cells, return_sequences=True, stateful=True, name='lsmt{}'.format(i+1))(previous_layer)
        lstms_inputs.append(lstm)
    output_dropout = Dropout(p=0.5)(lstms_inputs[-1])
    output = TimeDistributed(Dense(201, activation='softmax'), name='fc')(output_dropout)

    model = Model(input=input_features, output=output)
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print('Model Compiled!')

    h5_dataset = h5py.File(input_dataset, 'r')
    h5_predict = h5py.File(store_path, 'w')

    with open('dataset/videos.json', 'r') as f:
        videos_info = json.load(f)

    for subset in subsets:
        videos = [v for v in videos_info.keys() if videos_info[v]['subset'] == subset]
        videos = list(set(videos) & set(h5_dataset.keys()))
        nb_videos = len(videos)
        print('Predicting {} subset...'.format(subset))

        progbar = ProgressBar(max_value=nb_videos)
        count = 0
        output_subset = h5_predict.create_group(subset)
        for video_id in videos:
            progbar.update(count)
            video_features = h5_dataset[video_id][...]
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
    parser = argparse.ArgumentParser(description='Predict the output with the trained RNN')

    parser.add_argument('--id', dest='experiment_id', default=0, help='Experiment ID to track and not overwrite resulting models')

    parser.add_argument('-i', '--video-features', type=str, dest='video_features', default='data/dataset/video_features.hdf5', help='File where the video features are stored (default: %(default)s)')

    parser.add_argument('-n', '--num-cells', type=int, dest='num_cells', default=512, help='Number of cells for each LSTM layer when trained (default: %(default)s)')
    parser.add_argument('--num-layers', type=int, dest='num_layers', default=1, help='Number of LSTM layers of the network to train when trained (default: %(default)s)')

    parser.add_argument('-e', '--epoch', type=int, dest='epoch', default=100, help='epoch at which you want to load the weights from the trained model (default: %(default)s)')
    parser.add_argument('-o', '--output', type=str, dest='output_path', default='data/dataset', help='path to store the output file (default: %(default)s)')
    parser.add_argument('-s', '--subset', type=str, dest='subset', default=None, choices=['validation', 'testing'], help='Subset you want to predict the output (default: validation and testing)')

    args = parser.parse_args()

    extract_predicted_outputs(
        args.experiment_id,
        args.video_features,
        args.num_cells,
        args.num_layers,
        args.epoch,
        args.output_path,
        args.subset
    )
