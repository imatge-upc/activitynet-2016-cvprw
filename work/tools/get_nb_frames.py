import json
import sys

from progressbar import ProgressBar

import cv2
from work.config.config import STORED_VIDEOS_PATH


def get_number_of_frames(video_path):
    """ Returns the number of frames of the video stored at the path given
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open: " + video_path)
        return None

    return int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

def add_nb_frames_dataset(input_file, output_file):
    with open(input_file, 'r') as f:
        dataset = json.load(f)

    progbar = ProgressBar(max_value=len(dataset['database'].keys()))
    count = 0
    progbar.update(count)
    for video_id in dataset['database'].keys():
        nb_frames = get_number_of_frames(STORED_VIDEOS_PATH+'/'+video_id+'.mp4')
        dataset['database'][video_id].update({
            'num_frames': nb_frames
        })
        count += 1
        if count % 100 == 0:
            progbar.update(count)

    progbar.finish()

    with open(output_file, 'w') as f:
        json.dump(dataset, f)

if __name__ == '__main__':
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    add_nb_frames_dataset(input_file_path, output_file_path)
