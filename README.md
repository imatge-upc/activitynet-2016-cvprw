# UPC at ActivityNet Challenge 2016

This repository contains all the code related to the participation of the Universitat Politècnica
de Catalunya (UPC) at the ActivityNet Challenge 2016 at the CVPR.

All the code available is to reproduce and check the model proposed to face the classification and
detection task over the video dataset. It will be explained step by step all the stages required
to reproduce the results and also how to obtain predictions with our proposed model.


## Requirements

The first steps would require to have a Python
[virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/) set and work there.
It is recommended to use Python 2.7 as all the experiments have run over this version.

Next step is to install all the required packages:
```bash
virtualenv -p python2.7 --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

In addition to this, it will be required also to have OpenCV install on the machine to run it.
[Here](http://opencv.org/quickstart.html) there is the steps to install it. Assert that the
installation install the Python package. The installation can be on the machine as
`--system-site-packages` have been enabled, the cv package will be seen inside the virtual
environment.

All the experiments have been done with the Framework [Keras](http://keras.io/) with the
[Theano](https://github.com/Theano/Theano) as the computational backend. The version of Keras is a
fork with some modifications which allow to make the 3D operations from the C3D over GPU (the
original implementation crashed over GPU). To run it successfully over GPU, the file `~/.theanorc`
should look like this:
```
[global]
floatX = float32
device = gpu
optimizer_including = cudnn

[lib]
cnmem = 1

[dnn]
enabled = True
```

## Run Full Pipeline

To run the full pipeline, first it would be necessary to download the weights of both models: C3D and our trained model:

```bash
sh get_c3d_sports.sh
sh get_temporal_location_weights.sh

```

## Reproduce Experiments

### Download the ActivityNet v1.3 Dataset

The dataset is made up by videos from Youtube so they require to be download from the internet. To
do so, it has been used the [youtube-dl](https://rg3.github.io/youtube-dl/) package. To download all
the dataset, it has been extracted to the file `videos_ids.lst` the YouTube IDs of all the dataset
videos. Some of the videos are no longer available so they have been removed from the list, but
some others require to sign in to youtube to download it. For this reason, the download script will
require to give a valid YouTube login. Also, by default, all the videos will be downloaded into
the directory `./data/videos`. You can also specify which directory you want to store the videos.

```bash
cd dataset
# This will download the videos on the default directory
sh download_videos.sh username password
# This will download the videos on the directory you specify
sh download_videos.sh username password /path/you/want
```

### Extract the Features from the C3D Network

As the next step is to pass all the videos through the C3D network, first is required to download
the weights ported to [Keras](https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2).

```bash
cd data
sh get_c3d_sports.sh
```

Then, with the weights, there is a script which will read all the videos and extract its features.
To read the videos, it will require also to have OpenCV framework.

```bash
>> python -u features/extract_features.py -h
usage: extract_features.py [-h] [-d DIRECTORY] [-o OUTPUT] [-b BATCH_SIZE]
                           [-t NUM_THREADS] [-q QUEUE_SIZE] [-g NUM_GPUS]

Extract video features using C3D network

optional arguments:
  -h, --help            show this help message and exit
  -d DIRECTORY, --videos-dir DIRECTORY
                        videos directory (default: data/videos)
  -o OUTPUT, --output-dir OUTPUT
                        directory where to store the extracted features
                        (default: data/dataset)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size when extracting features (default: 32)
  -t NUM_THREADS, --num-threads NUM_THREADS
                        number of threads to fetch videos (default: 8)
  -q QUEUE_SIZE, --queue-size QUEUE_SIZE
                        maximum number of elements at the queue when fetching
                        videos (default 12)
  -g NUM_GPUS, --num-gpus NUM_GPUS
                        number of gpus to use for extracting features
                        (default: 1)
```

Because extracting a huge amount of features from a very big dataset (ActivityNet dataset videos have a 600GB size once downloaded) it require to do all the process very efficiently.

The script is based in producer/consumer paradigm, where there are multiple process fetching videos from disk (this task only requires CPU workload). Then one or multiple (not tested) process are created which each one works with one GPU and load the model and extract the features. Finally to safely store the extracted features, all the extracted ones are placed in a queue that a single process store them on a HDF5 file.

If appear any error trying to allocate memory from Theano, try to run over a GPU with a more memory, or reduce the batch size.
