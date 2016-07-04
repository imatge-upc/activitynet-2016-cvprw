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
cd data/models
sh get_c3d_sports.sh
sh get_temporal_location_weights.sh
```

Then it is only necessary to run the script with the input video specified:
```bash
python scripts/run_all_pipeline.py -i path/to/test/video.mp4
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
cd data/models
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

### Create Stateful Dataset

Once all the features have been extracted, it is required to place all the videos in batches but presenting continuity between them and so be able to train a recurrent neural network with a stateful approach.

![Stateful Dataset][stateful-dataset]

With the following script it will be created the stateful dataset for training and validation data and be stored in a HDF5 file:

```bash
>> python scripts/create_stateful_dataset.py -h
usage: create_stateful_dataset.py [-h] [-i VIDEO_FEATURES_FILE]
                                  [-v VIDEOS_INFO] [-l LABELS] [-o OUTPUT_DIR]
                                  [-b BATCH_SIZE] [-t TIMESTEPS]
                                  [-s {training,validation}]

Put all the videos features into the correct way to train a RNN in a stateful
way

optional arguments:
  -h, --help            show this help message and exit
  -i VIDEO_FEATURES_FILE, --video-features VIDEO_FEATURES_FILE
                        HDF5 where the video features have been extracted
                        (default: data/dataset/video_features.hdf5)
  -v VIDEOS_INFO, --videos-info VIDEOS_INFO
                        File containing the annotations of all the videos on
                        the dataset (default: dataset/videos.json)
  -l LABELS, --labels LABELS
                        File containing the labels of the whole dataset
                        (default: dataset/labels.txt)
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        directory where to store the stateful dataset
                        (default: data/dataset)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size desired to use for training (default: 256)
  -t TIMESTEPS, --timesteps TIMESTEPS
                        timesteps desired for training the RNN (default: 20)
  -s {training,validation}, --subset {training,validation}
                        Subset you want to create the stateful dataset
                        (default: training and validation)
```

### Train

The next step is to train the RNN using the provided script. The script also allows to change the configuration such as the learning rate, the number of LSTM cells or even the number of layers. During training, snapshots of the model weight's will be being stored for future prediction of the best model.

On `src/visualize` there is a function to plot the script's training.

```bash
>> python scripts/train.py -h
usage: train.py [-h] [--id EXPERIMENT_ID] [-i INPUT_DATASET] [-n NUM_CELLS]
                [--num-layers NUM_LAYERS] [-p DROPOUT_PROBABILITY]
                [-b BATCH_SIZE] [-t TIMESTEPS] [-e EPOCHS] [-l LEARNING_RATE]
                [-w LOSS_WEIGHT]

Train the RNN

optional arguments:
  -h, --help            show this help message and exit
  --id EXPERIMENT_ID    Experiment ID to track and not overwrite resulting
                        models
  -i INPUT_DATASET, --input-data INPUT_DATASET
                        File where the stateful dataset is stored (default:
                        data/dataset/dataset_stateful.hdf5)
  -n NUM_CELLS, --num-cells NUM_CELLS
                        Number of cells for each LSTM layer (default: 512)
  --num-layers NUM_LAYERS
                        Number of LSTM layers of the network to train
                        (default: 1)
  -p DROPOUT_PROBABILITY, --drop-prob DROPOUT_PROBABILITY
                        Dropout Probability (default: 0.5)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size used to create the stateful dataset
                        (default: 256)
  -t TIMESTEPS, --timesteps TIMESTEPS
                        timesteps used to create the stateful dataset
                        (default: 20)
  -e EPOCHS, --epochs EPOCHS
                        number of epochs to last the training (default: 100)
  -l LEARNING_RATE, --learning-rate LEARNING_RATE
                        learning rate for training (default: 1e-05)
  -w LOSS_WEIGHT, --loss-weight LOSS_WEIGHT
                        value to weight the loss to the background samples
                        (default: 0.3)
```

### Predict

Once the model is trained, its time to predict the results for the validation and test subset. To do so:

```bash
python scripts/predict.py -h
usage: predict.py [-h] [--id EXPERIMENT_ID] [-i VIDEO_FEATURES] [-n NUM_CELLS]
                  [--num-layers NUM_LAYERS] [-e EPOCH] [-o OUTPUT_PATH]
                  [-s {validation,testing}]

Predict the output with the trained RNN

optional arguments:
  -h, --help            show this help message and exit
  --id EXPERIMENT_ID    Experiment ID to track and not overwrite resulting
                        models
  -i VIDEO_FEATURES, --video-features VIDEO_FEATURES
                        File where the video features are stored (default:
                        data/dataset/video_features.hdf5)
  -n NUM_CELLS, --num-cells NUM_CELLS
                        Number of cells for each LSTM layer when trained
                        (default: 512)
  --num-layers NUM_LAYERS
                        Number of LSTM layers of the network to train when
                        trained (default: 1)
  -e EPOCH, --epoch EPOCH
                        epoch at which you want to load the weights from the
                        trained model (default: 100)
  -o OUTPUT_PATH, --output OUTPUT_PATH
                        path to store the output file (default: data/dataset)
  -s {validation,testing}, --subset {validation,testing}
                        Subset you want to predict the output (default:
                        validation and testing)
```

Be sure to specify correctly the `experiment_id` and the `epoch` of the previous trained model in order to use the correct weights.

### Post Processing

Finally, to obtain the classification and temporal localization of activities on the ActivityNet dataset, requires to do some post-processing. The script provided let choose some values but the default ones are the ones with better performance. The script returns 4 `json` files (classification and detection task for both validation and testing subset) with all the results in the format required by the ActivityNet Challenge.

```bash
>> python scripts/process_prediction.py -h
usage: process_prediction.py [-h] [--id EXPERIMENT_ID] [-p PREDICTIONS_PATH]
                             [-o OUTPUT_PATH] [-k SMOOTHING_K]
                             [-t ACTIVITY_THRESHOLD] [-s {validation,testing}]

Post-process the prediction of the RNN to obtain the classification and
temporal localization of the videos activity

optional arguments:
  -h, --help            show this help message and exit
  --id EXPERIMENT_ID    Experiment ID to track and not overwrite results
  -p PREDICTIONS_PATH, --predictions-path PREDICTIONS_PATH
                        Path where the predictions file is stored (default:
                        data/dataset)
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        Path where is desired to store the results (default:
                        data/dataset)
  -k SMOOTHING_K        Smoothing factor at post-processing (default: 5)
  -t ACTIVITY_THRESHOLD
                        Activity threshold at post-processing (default: 0.2)
  -s {validation,testing}, --subset {validation,testing}
                        Subset you want to post-process the output (default:
                        validation and testing)
```

[stateful-dataset]: https://raw.githubusercontent.com/imatge-upc/activitynet-2016-cvprw/master/misc/images/stateful_dataset.png
