import argparse
import os
import sys

import h5py
import numpy as np
import tensorflow as tf
import tflearn


def ctc_label_dense_to_sparse(labels, label_lengths):
    # undocumented feature soon to be made public
    from tensorflow.python.ops import functional_ops
    label_shape = tf.shape(labels)
    num_batches_tns = tf.pack([label_shape[0]])
    max_num_labels_tns = tf.pack([label_shape[1]])

    def range_less_than(previous_state, current_input):
        return tf.expand_dims(tf.range(label_shape[1]), 0) < current_input

    init = tf.cast(tf.fill(max_num_labels_tns, 0), tf.bool)
    dense_mask = functional_ops.scan(range_less_than, label_lengths,
                                     initializer=init, parallel_iterations=1)
    dense_mask = dense_mask[:, 0, :]

    label_array = tf.reshape(tf.tile(tf.range(0, label_shape[1]), num_batches_tns),
                             label_shape)
    label_ind = tf.boolean_mask(label_array, dense_mask)

    batch_array = tf.transpose(tf.reshape(tf.tile(tf.range(0, label_shape[0]),
                                                  max_num_labels_tns), tf.reverse(label_shape, [True])))
    batch_ind = tf.boolean_mask(batch_array, dense_mask)
    indices = tf.transpose(tf.reshape(tf.concat(0, [batch_ind, label_ind]), [2,-1]))

    vals_sparse = tf.gather_nd(labels, indices)

    return tf.SparseTensor(tf.to_int64(indices), vals_sparse, tf.to_int64(label_shape))


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape


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

    store_weights_root = 'model_snapshot_tf'
    store_weights_file = 'lstm_activity_detection_{experiment_id}_e{epoch:03}.hdf5'


    print('Loading Training Data...')
    f_dataset = h5py.File(input_dataset, 'r')
    X = f_dataset['training']['vid_features']
    Y = f_dataset['training']['output']
    Y = np.argmax(Y, axis=2).reshape(-1, timesteps, 1).astype(np.int32)
    # print('Loading Sample Weights...')
    # sample_weight = f_dataset['training']['sample_weight'][...]
    # sample_weight[sample_weight != 1] = loss_weight
    print('Loading Validation Data...')
    X_val = f_dataset['validation']['vid_features'][:5000]
    Y_val = f_dataset['validation']['output'][:5000]
    Y_val = np.argmax(Y_val, axis=2).reshape(-1, timesteps, 1).astype(np.int32)
    print('Loading Data Finished!')
    print('Input shape: {}'.format(X.shape))
    print('Output shape: {}\n'.format(Y.shape))
    print('Validation Input shape: {}'.format(X_val.shape))
    print('Validation Output shape: {}'.format(Y_val.shape))
    # print('Sample Weights shape: {}'.format(sample_weight.shape))
    Y = sparse_tuple_from(Y)
    Y_val = sparse_tuple_from(Y_val)


    print('Compiling model')
    input_ = tflearn.input_data(shape=[None, timesteps, 4096], name='input')
    target_ = tf.sparse_placeholder(tf.int32)
    # TODO: when feed the sparse placeholder try to pass a sparse represetation of the array:
    # https://github.com/igormq/ctc_tensorflow_example/blob/master/ctc_tensorflow_example.py#L26
    # Error: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/ctc_loss_op.cc#L62
    target = tflearn.input_data(shape=[None, timesteps], placeholder=target_, dtype=tf.int32, name='target')

    net = tflearn.batch_normalization(input_, name='normalization')
    for i in range(num_layers):
        net = tflearn.layers.recurrent.lstm(net, num_cells, return_seq=True, name='lstm_{}'.format(i+1))
    pred = tflearn.time_distributed(net, tflearn.fully_connected, [202, 'softmax'])

    pred = tf.transpose(pred, perm=[1, 0, 2])

    seq_length = tflearn.input_data(shape=[None], dtype=tf.int32)

    cost = tf.nn.ctc_loss(pred, target, sequence_length=seq_length)
    loss = tf.reduce_mean(cost)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)

    train_op = tflearn.TrainOp(loss, optimizer, batch_size=batch_size, shuffle=False)
    train = tflearn.Trainer([train_op], tensorboard_verbose=3, tensorboard_dir='log/lstm_training_tf', checkpoint_path=store_weights_root, random_seed=23)

    train_seq_length = np.array([timesteps]*X.shape[0], dtype=np.int32)
    val_seq_length = np.array([timesteps]*X_val.shape[0], dtype=np.int32)

    train.fit(
        feed_dicts={input_: X, seq_length: train_seq_length, target: Y},
        n_epoch=epochs,
        val_feed_dicts={input_: X_val, seq_length: val_seq_length, target: Y_val},
        snapshot_epoch=True,
        shuffle_all=False,
        run_id=experiment_id
    )


    print('Model Compiled!')


    trainer.fit(
        feed_dicts={input_: X, seq_length:train_seq_length, target: Y},
        n_epoch=epochs,
        val_feed_dicts={input_: X_val, seq_length:val_seq_length, target: Y_val},
        snapshot_epoch=True,
        shuffle_all=False,
        run_id=experiment_id
    )





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
