import json
import sys

import cv2

DOWNLOAD_PATH = '/imatge/amontes/work/datasets/ActivityNet/v1.3/videos'

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

    for video_id in dataset['database'].keys():
        nb_frames = get_number_of_frames(DOWNLOAD_PATH+'/'+video_id+'.mp4')
        dataset['database'][video_id].update({
            'num_frames': nb_frames
        })

    with open(output_file, 'w') as f:
        json.dump(dataset, f)

if __name__ == '__main__':
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    add_nb_frames_dataset(input_file_path, output_file_path)
