import json

import numpy as np

from work.dataset.activitynet import ActivityNetDataset
from work.processing.output import get_temporal_predictions

dataset = ActivityNetDataset(
    videos_path='../../dataset/videos.json',
    labels_path='../../dataset/labels.txt'
)
videos = dataset.get_subset_videos('validation')


with open('evaluation/data/result_template_validation.json', 'r') as f:
    results = json.load(f)
for video in videos:
    predictions_path = '../../downloads/predictions/v1/'+video.video_id+'.npy'
    output = np.load(predictions_path)
    predictions = get_temporal_predictions(output, fps=video.fps, clip_length=16)
    for p in predictions:
        label = dataset.labels[p['label']][1]
        p['label'] = label

    results['results'][video.video_id] = predictions

with open('evaluation/data/v1/detection.json', 'w') as f:
    json.dump(results, f)
