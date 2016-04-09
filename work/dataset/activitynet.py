import json

class ActivityNetDataset(object):

    def __init__(self, videos_path, labels_path, stored_videos_path=None):
        with open(videos_path, 'r') as dataset_file:
            database = json.load(dataset_file)
        self.database = database
        self.import_labels(labels_path)

        self.version = 'v1.3_cleaned'

        self.stored_videos_path = stored_videos_path

        self.videos = []
        for v_id in self.database.keys():
            self.videos.append(ActivityNetVideo(
                v_id,
                self.database[v_id],
                path=self.stored_videos_path
            ))

    def import_labels(self, labels_path):
        with open(labels_path, 'r') as f:
            lines = f.readlines()
        self.labels = []
        for l in lines:
            l = l.strip().split('\t')
            self.labels.append((l[0], l[1]))

    def get_subset(self, subset):
        """ Returns the videos corresponding to the given subset: training,
        validation or testing.
        """
        return {
            k: self.database for k in self.database.keys() \
                if self.database[k]['subset'] == subset
        }

    def get_labels(self):
        """ Returns the labels for all the videos
        """
        return [x[1] for x in self.labels]


    def get_stats(self):
        """ Return a descriptive stats of all the videos available in the
        dataset.
        """
        return {
            'videos': {
                'total': len(self.database.keys()),
                'training': len(self.get_subset('training').keys()),
                'validation': len(self.get_subset('validation').keys()),
                'testing': len(self.get_subset('testing').keys())
            },
            'labels': {
                'total': len(self.labels),
                'leaf_nodes': len(self.get_labels())
            }
        }

    def get_videos(self, subset):
        return [video for video in self.videos if video.subset == subset]

    def get_videos_from_label(self, label, input_videos=None):
        if input_videos is None:
            input_videos = self.videos
        return [video for video in input_videos if video.label == label]

    def get_total_duration(self):
        duration = 0
        for video in self.videos:
            duration += video.duration
        return duration

    def get_activity_duration(self, activity=None):
        videos = []
        if activity is None:
            videos = self.videos
        else:
            videos = self.get_videos_from_label(activity)

        duration = 0
        for video in videos:
            duration += video.get_activity_duration()
        return duration

class ActivityNetVideo:
    """ Class to encapsulate a video from the given dataset
    """
    def __init__(self, video_id, params, path=None, extension='mp4'):
        self.video_id = video_id
        self.url = params['url']
        self.subset = params['subset']
        self.resolution = params['resolution']
        self.duration = params['duration']
        self.annotations = params['annotations']
        self.label = None

        self.path = path
        self.extension = extension
        self.num_frames = params['num_frames']

        if self.annotations != []:
            self.label = self.annotations[0]['label']


    def get_activity(self):
        return self.label

    def get_activity_duration(self):
        duration = 0
        for annotation in self.annotations:
            duration += annotation['segment'][1] - annotation['segment'][0]
        return duration
