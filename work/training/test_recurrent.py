import os
import pickle
import sys

from work.config import RECURRENT_MODEL_WEIGHTS, STORED_DATASET_PATH
from work.dataset.activitynet import ActivityNetDataset
from work.models.decoder import RecurrentNetwork
from work.training.train_recurrent import load_data


def test():
    batch_size = 256
    timesteps = 20 # Entre 16 i 30

    # Loading dataset
    print('Loading dataset')
    if os.path.exists(STORED_DATASET_PATH):
        with open(STORED_DATASET_PATH, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = ActivityNetDataset(
            videos_path='../../dataset/videos.json',
            labels_path='../../dataset/labels.txt'
        )
        print('Generating Video Instances')
        dataset.generate_instances(length=16, overlap=0, subsets=('training', 'testing', 'validation'))
        with open(STORED_DATASET_PATH, 'wb') as f:
            pickle.dump(dataset, f)
    videos = dataset.get_subset_videos('validation')
    nb_instances = len(dataset.instances_validation)
    print('Number of instances: %d' % nb_instances)
    sys.stdout.flush()

    print('Compiling model')
    model = RecurrentNetwork(batch_size, timesteps, summary=True)
    model.load_weights(RECURRENT_MODEL_WEIGHTS)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print('Model Compiled!')

    print('Loading Data...')
    X, Y = load_data(videos, timesteps, batch_size)
    print('Loading Data Finished!')
    print('Input shape: {}'.format(X.shape))
    print('Output shape: {}\n'.format(Y.shape))
    sys.stdout.flush()
    history = model.evaluate(X, Y, batch_size=batch_size)
    print(history)
    sys.stdout.flush()


if __name__ == '__main__':
    test()
