import subprocess
import sys

INPUT_DIR = '/imatge/amontes/work/datasets/ActivityNet/subset'
OUTPUT_DIR = '/imatge/amontes/work/datasets/ActivityNet/train'

videos = []
with open('download_list_train.txt', 'r') as video_list:
    videos = video_list.readlines()

print('Total number of videos: {}'.format(len(videos)))
for video in videos:
    video = video.split('\n')[0]
    status = subprocess.call('mkdir {0}/{1}'.format(OUTPUT_DIR, video), shell=True)
    command = 'ffmpeg -i '+INPUT_DIR+'/'+video+'.mp4 '+OUTPUT_DIR+'/'+video+'/%05d.jpg'
    print(command)
    status = subprocess.call(command, shell=True)
    if status < 0:
        print('***************Video Frame Extraction Failed: {}***************'.format(video))
    else:
        print('Video frame extraction succeded: {}'.format(video))
