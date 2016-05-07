import json

import numpy as np

from work.dataset.activitynet import ActivityNetDataset
from work.processing.output import get_top_k_predictions_score

dataset = ActivityNetDataset(
    videos_path='../../dataset/videos.json',
    labels_path='../../dataset/labels.txt'
)
videos = dataset.get_subset_videos('validation')


with open('evaluation/data/result_template_validation.json', 'r') as f:
    results = json.load(f)
for video in videos:
    predictions_path = '../../downloads/predictions/v1/'+video.video_id+'.npy'
    predictions = np.load(predictions_path)
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

with open('evaluation/data/v1/classification.json', 'w') as f:
    json.dump(results, f)
