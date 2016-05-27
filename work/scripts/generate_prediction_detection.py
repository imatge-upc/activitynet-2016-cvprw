import json
import sys

import h5py
import numpy as np
from progressbar import ProgressBar

from work.dataset.activitynet import ActivityNetDataset
from work.processing.output import get_temporal_predictions_3


def main(predictions_path, output_file):
    dataset = ActivityNetDataset(
        videos_path='../../dataset/videos.json',
        labels_path='../../dataset/labels.txt'
    )

    f_predictions = h5py.File(predictions_path, 'r')
    for subset in ('validation',):
        print('Generating results for {} subset...'.format(subset))
        subset_predictions = f_predictions[subset]

        progbar = ProgressBar(max_value=len(subset_predictions.keys()))
        with open('evaluation/data/result_template_{}.json'.format(subset), 'r') as f:
            results = json.load(f)

        count = 0
        progbar.update(0)
        for video in dataset.get_subset_videos(subset):
            if video.video_id not in subset_predictions.keys():
                continue
            prediction = subset_predictions[video.video_id]
            class_predictions = np.argmax(prediction, axis=1)
            temporal_predictions = get_temporal_predictions_3(class_predictions, video.fps)
            for p in temporal_predictions:
                label = dataset.labels[p['label']][1]
                p['label'] = label

            results['results'][video.video_id] = temporal_predictions
            count += 1
            progbar.update(count)

        progbar.finish()
        with open(output_file.format(subset), 'w') as f:
            json.dump(results, f)

    f_predictions.close()


if __name__ == '__main__':
    predictions_path = str(sys.argv[1])
    output_file = str(sys.argv[2])
    main(predictions_path, output_file)
