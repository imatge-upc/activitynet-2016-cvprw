import json
import sys

import h5py
import numpy as np
from progressbar import ProgressBar

from work.dataset.activitynet import ActivityNetDataset


def main(predictions_path, ouptut_file):
    dataset = ActivityNetDataset(
        videos_path='../../dataset/videos.json',
        labels_path='../../dataset/labels.txt'
    )

    f_predictions = h5py.File(predictions_path, 'r')
    for subset in ('validation', 'testing'):
        print('Generating results for {} subset...'.format(subset))
        subset_predictions = f_predictions[subset]

        progbar = ProgressBar(max_value=len(subset_predictions.keys()))
        with open('evaluation/data/result_template_{}.json'.format(subset), 'r') as f:
            results = json.load(f)

        count = 0
        progbar.update(0)
        for video_id in subset_predictions.keys():
            predictions = subset_predictions[video_id]
            class_means = np.mean(predictions, axis=0)
            top_3 = np.argsort(class_means[1:])[::-1][:3] + 1
            scores = class_means[top_3]/np.sum(class_means[1:])
            result = []
            for index, score in zip(top_3, scores):
                label = dataset.labels[index][1]
                if score > 0:
                    result.append({
                        'score': score,
                        'label': label
                    })
            results['results'][video_id] = result
            count += 1
            progbar.update(count)
        progbar.finish()

        with open(output_file.format(subset), 'w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    predictions_path = str(sys.argv[1])
    output_file = str(sys.argv[2])
    main(predictions_path, output_file)
