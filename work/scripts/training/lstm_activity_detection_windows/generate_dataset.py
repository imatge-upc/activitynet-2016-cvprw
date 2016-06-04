import random

import h5py
import numpy as np
from progressbar import ProgressBar


def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def generate_dataset():
    f_video_features = h5py.File('/imatge/amontes/work/datasets/ActivityNet/v1.3/features_dataset/features_dataset.hdf5', 'r')
    f_audio_features = h5py.File('../dataset/audio_descriptors.hdf5', 'r')
    f_output = h5py.File('/imatge/amontes/work/datasets/ActivityNet/v1.3/output_dataset/output_dataset.hdf5', 'r')
    f_dataset = h5py.File('../dataset/dataset_for_detection_windows.hdf5', 'w')
    f_dataset_detection = f_dataset.create_group('detection')
    f_dataset_classification = f_dataset.create_group('classification')

    subsets = ('training', 'validation')

    batch_size = 256
    timesteps = 20
    overlap = .5
    threshold = .3

    features_size = 4096
    output_size = 200
    mfcc_size = 80
    spec_size = 8

    for subset in subsets:
        print('Subset {}'.format(subset))
        f_dataset_detection_subset = f_dataset_detection.create_group(subset)
        f_dataset_classification_subset = f_dataset_classification.create_group(subset)
        videos = f_video_features[subset].keys()

        videos = set(videos).intersection(f_audio_features['mfcc'].keys())
        videos = videos.intersection(f_audio_features['spec'].keys())
        videos = list(videos)
        random.shuffle(videos)

        nb_videos = len(videos)
        print('Number of videos: {}'.format(nb_videos))

        videos_shape = []
        for video_id in videos:
            videos_shape.append(f_video_features[subset][video_id].shape[0])
        nb_seqs = np.ceil(np.array(videos_shape, dtype=np.float32) / (timesteps*overlap)).astype(np.int64) - 1
        nb_seqs[nb_seqs == 0] = 1
        assert np.all(nb_seqs>0), nb_seqs

        total_seqs = np.sum(nb_seqs).astype(np.int64)

        video_features_detection = np.zeros((total_seqs, timesteps, features_size))
        mfcc_features_detection = np.zeros((total_seqs, timesteps, mfcc_size))
        spec_features_detection = np.zeros((total_seqs, timesteps, spec_size))
        output_detection = np.zeros((total_seqs/timesteps, 2))
        output_detection_classification = np.zeros((total_seqs/timesteps, output_size+1))
        video_features_classification = np.zeros((total_seqs, timesteps, features_size))
        mfcc_features_classification = np.zeros((total_seqs, timesteps, mfcc_size))
        spec_features_classification = np.zeros((total_seqs, timesteps, spec_size))
        output_classification = np.zeros((total_seqs/timesteps, output_size))

        progbar = ProgressBar(max_value=nb_videos)
        count = 0
        progbar.update(count)
        index_detection, index_classification = 0, 0
        for video_id, nb_seq in zip(videos, nb_seqs):
            vid_features = f_video_features[subset][video_id][...]
            assert vid_features.shape[1] == features_size
            nb_instances = vid_features.shape[0]
            # MFCC features
            mfcc_feat = f_audio_features['mfcc'][video_id][...]
            assert mfcc_feat.shape == (nb_instances, mfcc_size)
            # Spec features
            spec_feat = f_audio_features['spec'][video_id][...]
            assert spec_feat.shape == (1, spec_size), spec_feat.shape
            spec_feat = np.broadcast_to(spec_feat, (nb_instances, spec_size))
            # Output
            output_classes = f_output[subset][video_id][...]
            assert nb_instances == output_classes.shape[0]

            if nb_instances < timesteps:
                nb_repetitions = np.ceil(float(timesteps)/nb_instances).astype(np.int64)
                vid_features = np.tile(vid_features, (nb_repetitions, 1))[-timesteps:]
                mfcc_feat = np.tile(mfcc_feat, (nb_repetitions, 1))[-timesteps:]
                spec_feat = np.tile(spec_feat, (nb_repetitions, 1))[-timesteps:]
                output_classes = np.tile(output_classes, nb_repetitions)[-timesteps:]
                nb_instances = timesteps

            index = np.zeros((nb_seq, timesteps)).astype(np.int64)
            for i in range(nb_seq):
                s = i * timesteps * overlap
                index[i,:] = np.arange(s, s+timesteps)
            assert index[-1,-1] >= (nb_instances - 1), 'index[-1,-1] = {} and instances: {}'.format(index[-1,-1], nb_instances)
            dif = index[-1,-1] - nb_instances + 1
            index[-1,:] -= dif
            # if index[-2:-1] >= nb_instances:
            #     dif = index[-2,-1] - nb_instances + 1
            assert index[-1,-1] == (nb_instances - 1)

            print(output_classes)
            output_onehot = to_categorical(output_classes, nb_classes=(output_size+1))[index]
            output = output_onehot.mean(axis=1)
            assert output.shape == (nb_seq, output_size+1)
            activity_proportion = np.sum(output[:,1:], axis=1)
            assert output.shape[0] == activity_proportion.shape[0], 'output shape: {}, activity_proportion shape: {}'.format(output.shape, activity_proportion.shape)

            seq_with_activity = np.sum(activity_proportion>threshold)
            index_for_classification = index[activity_proportion>threshold,:]
            print(output[:,0])
            print(activity_proportion)
            print(output.shape)
            print(index_for_classification)
            print(nb_seq)
            print(seq_with_activity)
            print(output_detection[index_detection:index_detection+nb_seq, 1])

            # Store the values to the detection dataset
            video_features_detection[index_detection:(index_detection+nb_seq)] = vid_features[index]
            mfcc_features_detection[index_detection:(index_detection+nb_seq)] = mfcc_feat[index]
            spec_features_detection[index_detection:(index_detection+nb_seq)] = spec_feat[index]
            assert nb_seq == activity_proportion.shape[0]
            output_detection[index_detection:(index_detection+nb_seq), 0][(1-activity_proportion[:]) > threshold] = 1
            output_detection[index_detection:(index_detection+nb_seq), 1][activity_proportion[:] > threshold] = 1
            output_detection_classification[index_detection:(index_detection+nb_seq), :] = output_onehot

            # Store the values to the detection dataset
            video_features_classification[index_classification:index_classification+seq_with_activity] = vid_features[index_for_classification]
            mfcc_features_classification[index_classification:index_classification+seq_with_activity] = mfcc_feat[index_for_classification]
            spec_features_classification[index_classification:index_classification+seq_with_activity] = spec_feat[index_for_classification]
            output_classification[index_classification:index_classification+seq_with_activity] = output[activity_proportion > threshold,1:]

            # Update indexes
            index_detection += nb_seq
            index_classification += seq_with_activity

            count += 1
            progbar.update(count)

        index_detection = index_detection // 256 * 256
        index_classification = index_classification // 256 * 256
        video_features_detection = video_features_detection[:index_detection,:,:]
        mfcc_features_detection = mfcc_features_detection[:index_detection,:,:]
        spec_features_detection = spec_features_detection[:index_detection,:,:]
        output_detection = output_detection[:index_detection,:]

        video_features_classification = video_features_classification[:index_classification,:,:]
        mfcc_features_classification = mfcc_features_classification[:index_classification,:,:]
        spec_features_classification = spec_features_classification[:index_classification,:,:]
        output_classification = output_classification[:index_classification,:]

        assert np.all(np.any(video_features_detection, axis=2))
        assert np.all(np.any(mfcc_features_detection, axis=2))
        assert np.all(np.any(spec_features_detection, axis=2))
        assert np.all(np.any(output_detection, axis=1))
        assert np.all(np.any(video_features_classification, axis=2))
        assert np.all(np.any(mfcc_features_classification, axis=2))
        assert np.all(np.any(spec_features_classification, axis=2))
        assert np.all(np.any(output_classification, axis=1))

        # Save detection dataset
        f_dataset_detection_subset.create_dataset('video_features_detection',
            data=video_features_detection, chunks=(4, timesteps, features_size), dtype='float32')
        f_dataset_detection_subset.create_dataset('mfcc_features_detection',
            data=mfcc_features_detection, chunks=(128, timesteps, mfcc_size), dtype='float32')
        f_dataset_detection_subset.create_dataset('spec_features_detection',
            data=spec_features_detection, chunks=(batch_size, timesteps, spec_size), dtype='float32')
        f_dataset_detection_subset.create_dataset('output_detection',
            data=output_detection, chunks=(batch_size, 2), dtype='float32')

        # Save classification dataset
        f_dataset_classification_subset.create_dataset('video_features_classification',
            data=video_features_classification, chunks=(4, timesteps, features_size), dtype='float32')
        f_dataset_classification_subset.create_dataset('mfcc_features_classification',
            data=mfcc_features_classification, chunks=(128, timesteps, mfcc_size), dtype='float32')
        f_dataset_classification_subset.create_dataset('spec_features_classification',
            data=spec_features_classification, chunks=(batch_size, timesteps, spec_size), dtype='float32')
        f_dataset_classification_subset.create_dataset('output_classification',
            data=output_classification, chunks=(batch_size, output_size), dtype='float32')

        progbar.finish()

    print('Finish!!!')
    f_video_features.close()
    f_audio_features.close()
    f_output.close()
    f_dataset.close()

if __name__ == '__main__':
    generate_dataset()
