import argparse
import os
import sys

import h5py
import numpy as np
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Input, Lambda, TimeDistributed
from keras.models import Model
from keras.optimizers import RMSprop


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

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

    store_weights_root = './model_snapshot'
    store_weights_file = 'lstm_activity_classification_{experiment_id}_e{epoch:03}.hdf5'

    print('Loading Training Data...')
    f_dataset = h5py.File(input_dataset, 'r')
    X = f_dataset['training']['vid_features']
    Y = f_dataset['training']['output']
    # print('Loading Sample Weights...')
    # sample_weight = f_dataset['training']['sample_weight'][...]
    # sample_weight[sample_weight != 1] = loss_weight
    print('Loading Validation Data...')
    X_val = f_dataset['validation']['vid_features']
    Y_val = f_dataset['validation']['output']
    Y_labels = np.argmax(Y, axis=2)
    Y_val_labels = np.argmax(Y_val, axis=2)

    # Y_labels = Y_labels.reshape(-1, batch_size, timesteps)
    # Y_val_labels = Y_val_labels.reshape(-1, batch_size, timesteps)
    # Y_labels = Y_labels.reshape(-1, batch_size*timesteps)
    # Y_val_labels = Y_val_labels.reshape(-1, batch_size*timesteps)

    number_of_samples = X.shape[0]
    nb_batches = number_of_samples//batch_size + 1
    input_length = np.ones((number_of_samples,)) * timesteps
    #input_length[-1] -= number_of_samples - (nb_batches-1) * batch_size

    number_of_samples_val = X_val.shape[0]
    nb_batches_val = number_of_samples_val//batch_size + 1
    input_length_val = np.ones((number_of_samples_val,)) * timesteps
    #input_length_val[-1] -= number_of_samples_val - (nb_batches_val-1) * batch_size

    print('Loading Data Finished!')
    print('Input shape: {}'.format(X.shape))
    print('Output shape: {}\n'.format(Y_labels.shape))
    print('Validation Input shape: {}'.format(X_val.shape))
    print('Validation Output shape: {}'.format(Y_val_labels.shape))
    print(input_length.shape)
    print(input_length_val.shape)

    print('Compiling model')
    input_features = Input(batch_shape=(batch_size, timesteps, 4096,), name='features')
    input_normalized = BatchNormalization(name='normalization')(input_features)
    lstms_inputs = [input_normalized]
    for i in range(num_layers):
        previous_layer = lstms_inputs[-1]
        lstm = LSTM(num_cells, return_sequences=True, stateful=True, name='lsmt{}'.format(i+1))(previous_layer)
        lstms_inputs.append(lstm)

    y_pred = TimeDistributed(Dense(202, activation='softmax'), name='fc')(lstms_inputs[-1])

    Model(input=input_features, output=y_pred).summary()

    labels = Input(name='the_labels', shape=[timesteps], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")([y_pred, labels, input_length, label_length])

    model = Model(input=[input_features, labels, input_length, label_length], output=[loss_out])
    rmsprop = RMSprop(lr=lr)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=rmsprop, metrics=['accuracy'])
    print('Model Compiled!')

    callbacks = [TensorBoard(
        log_dir='log/train_lstm',
        histogram_freq=True,
        write_graph=True
    )]

    number_of_samples = X.shape[0]
    nb_batches = number_of_samples//batch_size + 1
    input_length = np.ones((nb_batches,)) * timesteps
    input_length[-1] -= number_of_samples - (nb_batches-1) * batch_size

    for i in range(1, epochs+1):
        print('Epoch {}/{}'.format(i, epochs))
        model.fit({
                        'features': X,
                        'the_labels': Y_labels,
                        'input_length': input_length,
                        'label_length': input_length
                  },
                  np.zeros((number_of_samples,)),
                  batch_size=batch_size,
                  validation_data=({
                                  'features': X_val,
                                  'the_labels': Y_val_labels,
                                  'input_length': input_length_val,
                                  'label_length': input_length_val
                            }, np.zeros((number_of_samples_val,))),
                  verbose=2,
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

    parser.add_argument('-i', '--input-data', type=str, dest='input_dataset', default='../../data/dataset/dataset_stateful.hdf5', help='File where the stateful dataset is stored (default: %(default)s)')

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
