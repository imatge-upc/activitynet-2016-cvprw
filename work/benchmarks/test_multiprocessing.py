import multiprocessing
from keras.preprocessing.video import video_to_array
import numpy as np
import time
import threading
import sys

path = './'

videos = ['-YMpwZkNc2A.mp4', '1R25VGmqS9o.mp4', '3AWvyAJv20g.mp4', '3pBldeB3uaE.mp4']

def fetch_batch(path):
    t1 = time.time()
    batch_size = 32
    batch = np.zeros((batch_size, 3, 16, 112, 112))
    for i in range(batch_size):
        vid = video_to_array(path, resize=(112, 112), length=16, start_frame=i*16)
        batch[i] = vid
    t2 = time.time()
    print('Time to fetch one batch: {:.2f}'.format(t2-t1))
    sys.stdout.flush()
    return t2-t1

print('Fetching batches one after another')
total = 0
for i in range(4):
    video = videos[i]
    total += fetch_batch(path+video)
print('Total time to fetch 4 batches: {:.2f}'.format(total))
sys.stdout.flush()

print('Fetch batches with threads')
threads = []
for i in range(4):
    video = videos[i]
    threads.append(threading.Thread(target=fetch_batch, args=[path+video]))

t1 = time.time()
for t in threads:
    t.start()

for t in threads:
    t.join()
t2 = time.time()
print('Total time to fetch 4 batches with threads is {:.2f} seconds'.format(t2-t1))
sys.stdout.flush()

print('Fetch batches with multiprocess')
process = []
for i in range(4):
    video = videos[i]
    process.append(multiprocessing.Process(target=fetch_batch, args=[path+video]))

t1 = time.time()
for p in process:
    p.start()

for p in process:
    p.join()
t2 = time.time()
print('Total time to fetch 4 batches with process is {:.2f} seconds'.format(t2-t1))
sys.stdout.flush()
