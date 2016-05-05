import numpy as np

from work.dataset.activitynet import ActivityNetDataset
from work.processing.data import get_top_k_predictions

dataset = ActivityNetDataset(
    videos_path='../../dataset/videos.json',
    labels_path='../../dataset/labels.txt'
)
videos = dataset.get_subset_videos('validation')

total_nb = float(len(videos))
print('Total number of videos: {}'.format(total_nb))
correct = 0.
jumped = 0
for video in videos:
    predictions_path = '../../downloads/predictions/v1/'+video.video_id+'.npy'
    predictions = np.load(predictions_path)
    top_k = get_top_k_predictions(predictions, k=1)
    if not top_k:
        jumped += 1
        continue
    label = dataset.labels[top_k[0]][1]
    #print(label, video.label)
    if label == video.label:
        correct += 1.

accuracy = correct / total_nb
print('Accuracy: {:.2f}%'.format(accuracy*100))
print('Jumped videos: {}'.format(jumped))
