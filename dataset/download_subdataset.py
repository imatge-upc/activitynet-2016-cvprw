import json
import os
import random

from pytube import YouTube

with open('../dataset/activity_net.v1-2.min.json') as raw_file:
    dataset = json.load(raw_file)
database = dataset['database']
taxonomy = dataset['taxonomy']

EXTENSION = 'mp4'
RESOLUTION = '360p'

download_dir = '../downloads/subdataset2/training/'

subset_category_size = 200
subset_category_instances = 10
subset = 'training'

subdataset = {}

all_node_ids = [x["nodeId"] for x in taxonomy]
leaf_node_ids = []
for x in all_node_ids:
    is_parent = False
    for query_node in taxonomy:
        if query_node["parentId"]==x: is_parent = True
    if not is_parent: leaf_node_ids.append(x)
leaf_nodes = [x for x in taxonomy if x["nodeId"] in  leaf_node_ids]

#subset_nodes = random.sample(leaf_nodes, subset_category_size)
subset_nodes = leaf_nodes

def download_video(name, url, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists(os.path.join(save_path, '{}.{}'.format(name, EXTENSION))):
        print('{} already downloaded'.format(name))
        return True
    try:
        yt = YouTube(url)
        yt.set_filename(name)
        video = yt.get(EXTENSION, RESOLUTION)
        video.download(save_path)
        print('Video {} downloaded'.format(url))
        return True
    except Exception as e:
        print('Couldn\'t download video {}\tError:{}'.format(url, str(e)))
        return False

for node in subset_nodes:
    count = 0
    videos = [x for x in database.keys() if \
              database[x]['subset'] == subset and \
              random.choice(database[x]['annotations'])['label'] == node['nodeName']]
    random.shuffle(videos)
    print('Downloading node: {}'.format(node['nodeName']))
    print('Number of videos: {}'.format(len(videos)))
    while count < subset_category_instances:
        name = videos.pop()
        v = database[name]
        success = download_video(name, v['url'], download_dir)
        if success:
              count += 1
              subdataset.update({name: v})


with open('../dataset/subdataset2.json', 'w') as raw_output:
    json.dump(subdataset, raw_output)
