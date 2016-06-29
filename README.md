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

```
./extract_features.py 
