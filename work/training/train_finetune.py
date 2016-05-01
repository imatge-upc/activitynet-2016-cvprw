from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import (Convolution3D, MaxPooling3D,
                                        ZeroPadding3D)
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from work.dataset.activitynet import ActivityNetDataset
from work.training.generator import VideoDatasetGenerator


def get_model(summary=False):
    """ Return the Keras model of the network
    """
    model = Sequential()
    # 1st layer group
    model.add(Convolution3D(64, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv1',
                            subsample=(1, 1, 1),
                            input_shape=(3, 16, 112, 112),
                            trainable=False))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv2',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3a',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3b',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4a',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4b',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5a',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5b',
                            subsample=(1, 1, 1),
                            trainable=False))
    model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropadding'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool5'))
    model.add(Flatten(name='flatten'))
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6', trainable=False))
    model.add(Dropout(.5, name='do1'))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5, name='do2'))
    model.add(Dense(487, activation='softmax', name='fc8'))

    # Load weights
    model.load_weights('../../models/c3d/c3d-sports1M_weights.h5')

    for _ in range(3):
        model.layers.pop()

    # FC layers for training
    model.add(Dense(4096, activation='relu', name='fc7x'))
    model.add(Dropout(.5, name='do2x'))
    model.add(Dense(201, activation='softmax', name='fc8x'))

    if summary:
        print(model.summary())
    return model


def train():
    experiment_nb = 1

    dataset = ActivityNetDataset(
        videos_path='/imatge/amontes/work/activitynet/dataset/videos.json',
        labels_path='/imatge/amontes/work/activitynet/dataset/labels.txt',
        stored_videos_path='/imatge/amontes/work/datasets/ActivityNet/v1.3/videos',
        files_extension='mp4'
    )
    dataset.generate_instances()
    print('Length of the dataset instances: {}'.format(len(dataset.instances)))
    samples_per_epoch = len(dataset.instances_training)
    print('Length of the training dataset instances: {}'.format(samples_per_epoch))

    # this should be a dictionary
    class_weights = dataset.compute_class_weights()
    print('Computed class weights')

    model = get_model(summary=True)
    sgd = SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    print('Model compiled')

    # generator
    generator = VideoDatasetGenerator(
        samplewise_center=False,
        samplewise_std_normalization=False,
        rotation_range=0.,
        width_shift_range=0.,
        height_shift_range=0.,
        horizontal_flip=False,
        vertical_flip=False,
        temporal_flip=False
    )

    # callbacks
    checkpointer = ModelCheckpoint(
        filepath="../../models/training/finetune/"+
                 "{:02d}".format(experiment_nb)+
                 "weights.{epoch:02d}-{val_loss:.2f}.hdf5",
        verbose=1,
        save_best_only=False
    )

    nb_epochs = 2
    batch_size = 32
    # training:
    history = model.fit_generator(
        generator.flow(
            dataset,
            'training',
            size=(112, 112),
            length=16,
            channels=3,
            batch_size=batch_size,
            shuffle=True,
            seed=8924
        ),
        samples_per_epoch,
        nb_epochs,
        verbose=1,
        callbacks=[checkpointer],
        validation_data=generator.flow(
            dataset,
            'validation',
            size=(112, 112),
            length=16,
            channels=3,
            batch_size=batch_size,
            shuffle=True,
            seed=8924
        ),
        nb_val_samples=128,
        class_weight=class_weights,
        max_q_size=16,
        nb_worker=10
    )


if __name__ == '__main__':
    train()
