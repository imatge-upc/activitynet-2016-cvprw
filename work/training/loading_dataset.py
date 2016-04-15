from work.dataset.activitynet import ActivityNetDataset

dataset = ActivityNetDataset(
    videos_path='/imatge/amontes/work/activitynet/dataset/videos.json',
    labels_path='/imatge/amontes/work/activitynet/dataset/labels.txt',
    stored_videos_path='/imatge/amontes/work/datasets/ActivityNet/v1.3/videos'
)

print('Initial dataset instances:', dataset.instances)
dataset.generate_instances()

print('Length of the dataset instances:', len(dataset.instances))
print('Length of the training dataset instances:',  len(dataset.instances_training))
