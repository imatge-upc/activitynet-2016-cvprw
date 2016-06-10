import random

import h5py
import numpy as np
from progressbar import ProgressBar

f_video_features = h5py.File('/imatge/amontes/work/datasets/ActivityNet/v1.3/features_dataset/features_dataset.hdf5', 'r')
f_audio_features = h5py.File('./training/dataset/audio_descriptors.hdf5', 'r')
f_output = h5py.File('/imatge/amontes/work/datasets/ActivityNet/v1.3/output_dataset/output_dataset.hdf5', 'r')
f_dataset = h5py.File('training/dataset/stateful_dataset_with_audio_feedback.hdf5', 'w')

subsets = ('training', 'validation')
max_nb_videos = None

batch_size = 256
timesteps = 20

features_size = 4096
output_size = 201
mfcc_size = 80
spec_size = 8

def to_categorical(y, nb_classes=None):
    ''' Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

for subset in subsets:
    videos = f_video_features[subset].keys()

    videos = set(videos).intersection(f_audio_features['mfcc'].keys())
    videos = videos.intersection(f_audio_features['spec'].keys())
    videos = list(videos)

    random.shuffle(videos)
    if max_nb_videos and max_nb_videos < len(videos):
        videos = videos[:max_nb_videos]

    nb_videos = len(videos)
    print('Number of videos: {}'.format(nb_videos))

    sequence_stack = []
    for _ in range(batch_size):
        sequence_stack.append([])
    nb_clips_stack = np.zeros(batch_size).astype(np.int64)
    accumulative_clips_stack = []
    for _ in range(batch_size):
        accumulative_clips_stack.append([])

    for video_id in videos:
        min_pos = np.argmin(nb_clips_stack)
        sequence_stack[min_pos].append(video_id)
        nb_clips_stack[min_pos] += f_video_features[subset][video_id].shape[0]
        accumulative_clips_stack[min_pos].append(nb_clips_stack[min_pos])

    min_sequence = np.min(nb_clips_stack)
    max_sequence = np.max(nb_clips_stack)
    nb_batches_long = max_sequence // timesteps + 1
    nb_batches = min_sequence // timesteps
    print('Number of batches: {}'.format(nb_batches))

    video_features = np.zeros((nb_batches_long*batch_size*timesteps, features_size))
    mfcc_features = np.zeros((nb_batches_long*batch_size*timesteps, mfcc_size))
    spec_features = np.zeros((nb_batches_long*batch_size*timesteps, spec_size))
    output = np.zeros((nb_batches_long*batch_size*timesteps, output_size))
    prev_out = np.zeros((nb_batches_long*batch_size*timesteps, output_size+1))
    index = np.arange(nb_batches_long*batch_size*timesteps)

    progbar = ProgressBar(max_value=batch_size)

    for i in range(batch_size):
        batch_index = index // timesteps % batch_size == i
        progbar.update(i)

        pos = 0
        for video_id in sequence_stack[i]:
            # Video features
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


            video_index = index[batch_index][pos:pos+nb_instances]
            video_features[video_index,:] = vid_features
            mfcc_features[video_index,:] = mfcc_feat
            spec_features[video_index,:] = spec_feat
            output[video_index,:] = to_categorical(output_classes, nb_classes=output_size)
            prev_output = np.zeros((nb_instances, output_size+1))
            prev_output[0,-1] = 1
            prev_output[1:,:output_size] = to_categorical(output_classes, nb_classes=output_size)[:-1,:]
            prev_out[video_index,:] = prev_output

            pos += nb_instances

    progbar.finish()

    video_features = video_features[:nb_batches*batch_size*timesteps,:]
    assert np.all(np.any(video_features, axis=1))
    video_features = video_features.reshape((nb_batches*batch_size, timesteps, features_size))

    mfcc_features = mfcc_features[:nb_batches*batch_size*timesteps,:]
    assert np.all(np.any(mfcc_features, axis=1))
    mfcc_features = mfcc_features.reshape((nb_batches*batch_size, timesteps, mfcc_size))

    spec_features = spec_features[:nb_batches*batch_size*timesteps,:]
    assert np.all(np.any(spec_features, axis=1))
    spec_features = spec_features.reshape((nb_batches*batch_size, timesteps, spec_size))

    output = output[:nb_batches*batch_size*timesteps,:]
    assert np.all(np.any(output, axis=1))
    output = output.reshape((nb_batches*batch_size, timesteps, output_size))

    prev_out = prev_out[:nb_batches*batch_size*timesteps,:]
    assert np.all(np.any(prev_out, axis=1))
    prev_out = prev_out.reshape((nb_batches*batch_size, timesteps, output_size+1))

    if subset == 'training':
        background_weight = 0.6
        sample_weights = np.ones(output.shape[:2])
        sample_weights[output[:,:,0] == 1] = background_weight

    f_dataset_subset = f_dataset.create_group(subset)

    f_dataset_subset.create_dataset('vid_features', data=video_features, chunks=(4, timesteps, features_size), dtype='float32')
    f_dataset_subset.create_dataset('mfcc_features', data=mfcc_features, chunks=(256, timesteps, mfcc_size), dtype='float32')
    f_dataset_subset.create_dataset('spec_features', data=spec_features, chunks=(256, timesteps, spec_size), dtype='float32')
    f_dataset_subset.create_dataset('output', data=output, chunks=(256, timesteps, output_size), dtype='float32')
    f_dataset_subset.create_dataset('prev_output', data=prev_out, chunks=(256, timesteps, output_size+1), dtype='float32')
    if subset == 'training':
        f_dataset_subset.create_dataset('sample_weight', data=sample_weights, chunks=(256, timesteps), dtype='float32')

f_video_features.close()
f_audio_features.close()
f_output.close()
f_dataset.close()
