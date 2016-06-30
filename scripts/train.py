import argparse
import os
import sys

import h5py

from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop


def train(experiment_id, input_dataset, num_cells, num_layers, dropout_probability, batch_size, timesteps, epochs, lr, loss_weight):
    print('Experiment ID {}'.format(experiment_id))

    print('number of cells: {}'.format(num_cells))
    print('number of layers: {}'.format(num_layers))
    print('dropout probability: {}'.format(dropout_probability))

    print('batch_size: {}'.format(batch_size))
    print('timesteps: {}'.format(timesteps))
    print('epochs: {}'.format(epochs))
    print('learning rate: {}'.format(lr))
    print('loss weight for background class: {}\n'.format(loss_weight))

    store_weights_root = 'data/model_snapshot'
    store_weights_file = 'lstm_activity_classification_{experiment_id}_e{epoch:03}.hdf5'

    print('Compiling model')
    input_features = Input(batch_shape=(batch_size, timesteps, 4096,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    input_dropout = Dropout(p=dropout_probability)(input_normalized)
    lstms_inputs = [input_dropout]
    for i in range(num_layers):
        previous_layer = lstms_inputs[-1]
        lstm = LSTM(num_cells, return_sequences=True, stateful=True, name='lsmt{}'.format(i+1))(previous_layer)
        lstms_inputs.append(lstm)

    output_dropout = Dropout(p=dropout_probability)(lstms_inputs[-1])
    output = TimeDistributed(Dense(201, activation='softmax'), name='fc')(output_dropout)

    model = Model(input=input_features, output=output)
    model.summary()
    rmsprop = RMSprop(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'],
        sample_weight_mode='temporal')
    print('Model Compiled!')

    print('Loading Training Data...')
    f_dataset = h5py.File(input_dataset, 'r')
    X = f_dataset['training']['vid_features']
    Y = f_dataset['training']['output']
    print('Loading Sample Weights...')
    sample_weight = f_dataset['training']['sample_weight'][...]
    sample_weight[sample_weight != 1] = loss_weight
    print('Loading Validation Data...')
    X_val = f_dataset['validation']['vid_features']
    Y_val = f_dataset['validation']['output']
    print('Loading Data Finished!')
    print('Input shape: {}'.format(X.shape))
    print('Output shape: {}\n'.format(Y.shape))
    print('Validation Input shape: {}'.format(X_val.shape))
    print('Validation Output shape: {}'.format(Y_val.shape))
    print('Sample Weights shape: {}'.format(sample_weight.shape))

    for i in range(1, epochs+1):
        print('Epoch {}/{}'.format(i, epochs))
        model.fit(X,
                  Y,
                  batch_size=batch_size,
                  validation_data=(X_val, Y_val),
                  sample_weight=sample_weight,
                  verbose=1,
                  nb_epoch=1,
                  shuffle=False)
        print('Reseting model states')
        model.reset_states()
        if (i % 5) == 0:
            print('Saving snapshot...')
            save_name = store_weights_file.format(experiment_id=experiment_id, epoch=i)
            save_path = os.path.join(store_weights_root, save_name)
            model.save_weights(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the RNN ')

    parser.add_argument('--id', dest='experiment_id', default=0, help='Experiment ID to track and not overwrite resulting models')

    parser.add_argument('-i', '--input-data', type=str, dest='input_dataset', default='data/dataset/dataset_stateful.hdf5', help='File where the stateful dataset is stored (default: %(default)s)')

    parser.add_argument('-n', '--num-cells', type=int, dest='num_cells', default=512, help='Number of cells for each LSTM layer (default: %(default)s)')
    parser.add_argument('--num-layers', type=int, dest='num_layers', default=1, help='Number of LSTM layers of the network to train (default: %(default)s)')
    parser.add_argument('-p', '--drop-prob', type=float, dest='dropout_probability', default=.5, help='Dropout Probability (default: %(default)s)')

    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size', default=256, help='batch size used to create the stateful dataset (default: %(default)s)')
    parser.add_argument('-t', '--timesteps', type=int, dest='timesteps', default=20, help='timesteps used to create the stateful dataset (default: %(default)s)')
    parser.add_argument('-e', '--epochs', type=int, dest='epochs', default=100, help='number of epochs to last the training (default: %(default)s)')
    parser.add_argument('-l', '--learning-rate', type=float, dest='learning_rate', default=1e-5, help='learning rate for training (default: %(default)s)')
    parser.add_argument('-w', '--loss-weight', type=float, dest='loss_weight', default=.3, help='value to weight the loss to the background samples (default: %(default)s)')

    args = parser.parse_args()

    train(
        args.experiment_id,
        args.input_dataset,
        args.num_cells,
        args.num_layers,
        args.dropout_probability,
        args.batch_size,
        args.timesteps,
        args.epochs,
        args.learning_rate,
        args.loss_weight
    )
