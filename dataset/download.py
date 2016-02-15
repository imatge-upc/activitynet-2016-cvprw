import json
import logging
import requests
import os

from pytube import YouTube

EXTENSION = 'mp4'       # Extension to download the video (other option '.3gp')
RESOLUTION = '360p'     # Resolution to download the video (other option '240p')

LOGGER = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(__file__)

def download_dataset(
        download_dir='../downloads/dataset/',
        version='1.2',
        subset=None
    ):
    """
    Function to download all the dataset from the ActivityNet dataset
    """
    if version == '1.1':
        file_name = 'activity_net.v1-1.min.json'
    elif version == '1.2':
        file_name = 'activity_net.v1-2.min.json'
    else:
        raise Exception('Version Not Available')

    with open(os.path.join(BASE_DIR, file_name)) as raw_file:
        dataset = json.load(raw_file)

    database = dataset['database']
    total_size = 0
    for key, video in database.items():
        if subset is not None and subset != video['subset']:
            continue
        save_path = os.path.join(download_dir, video['subset'])
        #download_video(key, video['url'], save_path)
        total_size += get_size(key, video['url'])

def download_video(name, url, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists(os.path.join(save_path, '{}.{}'.format(name, EXTENSION))):
        LOGGER.info('{} already downloaded'.format(name))
        return

    try:
        yt = YouTube(url)
        yt.set_filename(name)
        video = yt.get(EXTENSION, RESOLUTION)
        video.download(save_path)
        LOGGER.info('Video {} downloaded'.format(url))
    except Exception as e:
        LOGGER.error(str(e))
        LOGGER.error('Couldn\'t download video {}'.format(url))

def get_size(key, url):
    file_size = 0
    try:
        yt = YouTube(url)
        video = yt.get(EXTENSION, RESOLUTION)
        url = video.url
        while file_size == 0:
            response = requests.head(url)
            file_size = int(response.headers.get("Content-Length") or
                            response.headers.get("content-length"))
            if file_size == 0:
                url = response.headers['location']
        print('{}: {:.2f} MBytes'.format(key, file_size/(1024*1024)))
    except Exception as e:
        LOGGER.error(str(e))
        LOGGER.error('Couldn\'t get video  size from {}'.format(url))
    return file_size


if __name__ == '__main__':
    download_dataset()
