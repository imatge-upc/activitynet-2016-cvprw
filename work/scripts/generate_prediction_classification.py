import json
import sys

import numpy as np

from work.dataset.activitynet import ActivityNetDataset
from work.processing.output import get_top_k_predictions_score
from work.tools.utils import get_files_in_dir


def main(predictions_path, ouptut_file):
    dataset = ActivityNetDataset(
        videos_path='../../dataset/videos.json',
        labels_path='../../dataset/labels.txt'
    )
    extracted_features = get_files_in_dir('../../downloads/predictions/lstm_activity_classification/v1/classes', extension='npy')
    # Remove the videos which features hasn't been extracted
    videos_to_remove = []
    for video in dataset.videos:
        if video.video_id not in extracted_features:
            videos_to_remove.append(video)
    for video in videos_to_remove:
        dataset.videos.remove(video)

    videos = dataset.get_subset_videos('validation')


    with open('evaluation/data/result_template_validation.json', 'r') as f:
        results = json.load(f)
    for video in videos:
        prediction_path = predictions_path+video.video_id+'.npy'
        predictions = np.load(prediction_path)
        top_k, scores = get_top_k_predictions_score(predictions, k=3)
        result = []
        for index, score in zip(top_k, scores):
            label = dataset.labels[index][1]
            if score > 0:
                result.append({
                    'score': score,
                    'label': label
                })
        results['results'][video.video_id] = result
        print(video.video_id, top_k)

    with open(output_file, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    predictions_path = str(sys.argv[1])
    output_file = str(sys.argv[2])
    main(predictions_path, output_file)
