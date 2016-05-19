""" Read all the dataset from the extracted features and store it in a hdf5 dataset """
import os

import h5py
import numpy as np
from progressbar import ProgressBar

from work.dataset.activitynet import ActivityNetDataset
from work.environment import (DATASET_LABELS, DATASET_VIDEOS,
                              FEATURES_DATASET_FILE,
                              FEATURES_MASKED_DATASET_FILE,
                              OUTPUTS_DATASET_FILE,
                              OUTPUTS_MASKED_DATASET_FILE, STORED_DATASET_PATH,
                              STORED_DATASET_PATH_2, STORED_FEATURES_PATH)
from work.tools.utils import get_files_in_dir

try:
    import cPickle as pickle
except:
    import pickle


def import_dataset():
    print('Loading dataset')

    dataset = ActivityNetDataset(
        videos_path=DATASET_VIDEOS,
        labels_path=DATASET_LABELS
    )

    extracted_features = get_files_in_dir(STORED_FEATURES_PATH, extension='npy')
    # Remove the videos which features hasn't been extracted
    videos_to_remove = []
    for video in dataset.videos:
        if video.video_id not in extracted_features:
            videos_to_remove.append(video)
    for video in videos_to_remove:
        dataset.videos.remove(video)

    print('Generating Video Instances')
    dataset.generate_instances(length=16, overlap=0, subsets=('training', 'validation'))

    # # For features data
    # f_features = h5py.File(FEATURES_DATASET_FILE, 'w')
    # for subset in ('training', 'validation', 'testing'):
    #     videos = dataset.get_subset_videos(subset)
    #     nb_videos = len(videos)
    #     subset_group = f_features.create_group(subset)
    #
    #     print('Reading features for {} subset'.format(subset))
    #     progbar = ProgressBar(max_value=nb_videos)
    #     count = 0
    #     for video in videos:
    #         progbar.update(count)
    #         features_path = os.path.join(STORED_FEATURES_PATH, video.features_file_name)
    #         features = np.load(features_path)
    #         subset_group.create_dataset(video.video_id, data=features)
    #         count += 1
    #     progbar.finish()
    # f_features.close()

    # For output data
    f_outputs = h5py.File(OUTPUTS_DATASET_FILE, 'w')
    for subset in ('training', 'validation'):
        videos = dataset.get_subset_videos(subset)
        nb_videos = len(videos)
        subset_group = f_outputs.create_group(subset)

        print('Reading outputs for {} subset'.format(subset))
        progbar = ProgressBar(max_value=nb_videos)
        count = 0
        for video in videos:
            progbar.update(count)
            outputs = [instance.output for instance in video.instances]
            outputs = np.array(outputs, dtype=np.int16)
            subset_group.create_dataset(video.video_id, data=outputs)
            count += 1
        progbar.finish()
    f_outputs.close()

def import_dataset_masking():
    print('Loading dataset')

    dataset = ActivityNetDataset(
        videos_path=DATASET_VIDEOS,
        labels_path=DATASET_LABELS
    )

    extracted_features = get_files_in_dir(STORED_FEATURES_PATH, extension='npy')
    # Remove the videos which features hasn't been extracted
    videos_to_remove = []
    for video in dataset.videos:
        if video.video_id not in extracted_features:
            videos_to_remove.append(video)
    for video in videos_to_remove:
        dataset.videos.remove(video)

    print('Generating Video Instances')
    dataset.generate_instances(length=16, overlap=0, subsets=('training', 'validation'))

    f_input = h5py.File(FEATURES_MASKED_DATASET_FILE, 'w')
    data = f_input.create_group('dataset')
    videos_ids = f_input.create_group('videos_ids')
    for subset in ('training', 'validation', 'testing'):
        videos = dataset.get_subset_videos(subset)
        nb_videos = len(videos)
        subset_data = data.create_dataset(subset, (nb_videos, 500, 4096), dtype='float32')
        subset_ids = videos_ids.create_dataset(subset, (nb_videos,), dtype='S11')

        print('Reading outputs for {} subset'.format(subset))
        progbar = ProgressBar(max_value=nb_videos)
        pos = 0
        for video in videos:
            progbar.update(pos)
            features_path = os.path.join(STORED_FEATURES_PATH, video.features_file_name)
            features = np.load(features_path)
            nb_instances = features.shape[0]
            if nb_instances <= 500:
                subset_data[pos,:nb_instances,:] = features
                subset_ids[pos] = np.string_(video.video_id)
            elif nb_instances <= 1000:
                subset_data[pos,:500,:] = features[:500,:]
                subset_ids[pos] = np.string_(video.video_id)
                pos += 1
                subset_data[pos,500:,:] = features[500:,:]
                subset_ids[pos] = np.string_(video.video_id)
                pos += 1
            elif nb_instances <= 1500:
                subset_data[pos,:500,:] = features[:500,:]
                subset_ids[pos] = np.string_(video.video_id)
                pos += 1
                subset_data[pos,500:1000,:] = features[500:1000,:]
                subset_ids[pos] = np.string_(video.video_id)
                pos += 1

            pos += 1
        progbar.finish()
    f_input.close()

if __name__ == '__main__':
    import_dataset_masking()
