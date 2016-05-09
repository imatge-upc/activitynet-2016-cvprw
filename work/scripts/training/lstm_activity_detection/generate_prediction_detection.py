import json

import numpy as np

from work.dataset.activitynet import ActivityNetDataset
from work.processing.output import get_temporal_predictions_3
from work.tools.utils import get_files_in_dir


def main():
    detection_predictions_path = '../../../../downloads/predictions/lstm_activity_detection/v1/classes/'
    classification_predictions_path = '../../../../downloads/predictions/lstm_activity_classification/v1/classes/'

    dataset = ActivityNetDataset(
        videos_path='../../../../dataset/videos.json',
        labels_path='../../../../dataset/labels.txt'
    )
    extracted_features = get_files_in_dir('../../../../downloads/predictions/lstm_activity_classification/v1/classes', extension='npy')
    # Remove the videos which features hasn't been extracted
    videos_to_remove = []
    for video in dataset.videos:
        if video.video_id not in extracted_features:
            videos_to_remove.append(video)
    for video in videos_to_remove:
        dataset.videos.remove(video)

    videos = dataset.get_subset_videos('validation')

    # Load the results templates
    with open('../../evaluation/data/result_template_validation.json', 'r') as f:
        results = json.load(f)

    for video in videos:
        detection_prediction_path = detection_predictions_path + video.video_id + '.npy'
        classification_prediction_path =  classification_predictions_path + video.video_id + '.npy'

        class_prediction = np.load(classification_prediction_path)
        detection_prediction = np.load(detection_prediction_path)

        mix = np.zeros(class_prediction.shape, dtype=np.int64)
        for pos in range(class_prediction.size):
            if detection_prediction[pos] == 1:
                mix[pos] = class_prediction[pos]

        predictions = get_temporal_predictions_3(mix, fps=video.fps, k=1)
        for p in predictions:
            label = dataset.labels[p['label']][1]
            p['label'] = label

        results['results'][video.video_id] = predictions

    with open('../../evaluation/data/results/activity_detection/v1/detection_3.json', 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()
