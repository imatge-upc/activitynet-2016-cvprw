import os
import random

import numpy as np

import cv2


def get_sample_frame_from_video(videoid, duration, start_time, end_time,
                                video_path):
    filename = os.path.join(video_path, '{}.mp4'.format(videoid))
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print("Could not open: " + video_path)
        return None
    nr_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    fps = (nr_frames*1.0)/duration

    start_frame, end_frame = int(start_time*fps), int(end_time*fps)
    frame_idx = random.choice(range(start_frame, end_frame))
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_idx)
    ret, img = cap.read()
    if not ret:
        raise Exception('Could not read frame {} for video {}'.format(frame_idx, videoid))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
