# Temporal Activity Detection in Untrimmed Videos with Recurrent Neural Networks

This project page describes our paper at the [1st NIPS Workshop on Large Scale Computer Vision Systems](https://sites.google.com/site/largescalecvsystems/cfp/index). This work also corresponds to the submission of the UPC team participating in the [ActivityNet Challenge][activitynet-challenge] for CVPR 2016.

| ![Alberto Montes][image-alberto] | ![Amaia Salvador][image-amaia] | ![Xavier Giró-i-Nieto][image-xavier] | ![Santiago Pascual][image-santi] |
| :---: | :---: | :---: | :---: |
| Main contributor | Advisor | Advisor | Co-advisor |
| [Alberto Montes][web-alberto] | [Amaia Salvador][web-amaia] | [Xavier Giró-i-Nieto][web-xavier] | [Santiago Pascual][web-santi] |

Institution: [Universitat Politècnica de Catalunya](http://www.upc.edu).

![Universitat Politècnica de Catalunya][image-upc-logo]


## Abstract

This work proposes a simple pipeline to classify and temporally localize activities in untrimmed videos. Our system uses features from a 3D Convolutional Neural Network (C3D) as input to train a a recurrent neural network (RNN) that learns to classify video clips of 16 frames. After clip prediction, we post-process the output of the RNN to assign a single activity label to each video, and determine the temporal boundaries of the activity within the video. We show how our system can achieve competitive results in both tasks with a simple architecture. We evaluate our method in the ActivityNet Challenge 2016, achieving a 0.5874 mAP and a 0.2237 mAP in the classification and detection tasks, respectively.

## What Are You Going to Find Here

This project is a baseline in the activity classification and its temporal location, focused on the [ActivityNet Challenge][activitynet-challenge]. Here is detailed all the process of our proposed pipeline, as well the trained models and the utility to classify and temporally localize activities on new videos given. All the steps have been detailed, from downloading the dataset, to predicting the temporal locations going through the feature extraction and also the training.

## Publication

Download our paper at the [1st NIPS Workshop on Large Scale Computer Vision Systems](https://sites.google.com/site/largescalecvsystems/cfp/index) by clicking [here](https://github.com/imatge-upc/activitynet-2016-cvprw/raw/master/temporal-activity-detection.pdf). Please cite with the following Bibtex code:

```
@InProceedings{Montes_2016_NIPSWS,
author = {Montes, Alberto and Salvador, Amaia and Pascual, Santiago and Giro-i-Nieto, Xavier},
title = {Temporal Activity Detection in Untrimmed Videos with Recurrent Neural Networks},
booktitle = {1st NIPS Workshop on Large Scale Computer Vision Systems},
month = {December},
year = {2016}
}
```

You may also want to refer to our publication with the more human-friendly Chicago style:

*Alberto Montes, Amaia Salvador, Santiago Pascual, and Xavier Giro-i-Nieto. "Temporal Activity Detection in Untrimmed Videos with Recurrent Neural Networks." In 1st NIPS Workshop on Large Scale Computer Vision Systems. 2016.*

This work is the result of the bachelor thesis by Alberto Montes at UPC TelecomBCN ETSETB during Spring 2016. Please check his [technical report](https://arxiv.org/abs/1608.08128), [slides](http://www.slideshare.net/xavigiro/temporal-activity-detection-in-untrimmed-videos-with-recurrent-neural-networks?ref=https://imatge-upc.github.io/activitynet-2016-cvprw/) and [oral presentation](https://www.youtube.com/watch?v=3G-Vdmsluw0) for more details.

## Repository Structure

This repository is structured in the following way:
* `data/`: dir where, by default, all the data such as videos or model weights are stored. Some data is given such ass the C3D means and also provide scripts to download the weights for the C3D model and the one we propose.
* `dataset/`: files describing the ActivityNet dataset and a script to download all the videos. The information of the dataset has been extended with the number of frames at each of the videos.
* `misc/`: directory with some miscellaneous information such as all the details of the steps followed on this project and much more.
* `notebooks/`: notebooks with some visualization of the results.
* `scripts/`: scripts to reproduce all the steps of project.
* `src/`: source code required for the scripts.

## Dependencies

This project is build using the [Keras](https://github.com/fchollet/keras) library for Deep Learning, which can use as a backend both [Theano](https://github.com/Theano/Theano)
and [TensorFlow](https://github.com/tensorflow/tensorflow).

We have used Theano in order to develop the project because it supported 3D convolutions and pooling required to run the C3D network.

For a further and more complete of all the dependencies used within this project, check out the requirements.txt provided within the project. This file will help you to recreate the exact same Python environment that we worked with.

## Pipeline

The pipeline proposed to face the ActivityNet Challenge is made up by two stages.

The first stage encode the video information into a single vector representation for small video clips. To achieve that, the C3D network [Tran2014] is used. The C3D network uses 3D convolutions to extract spatiotemporal features from the videos, which previously have been split in 16-frames clips.

The second stage, once the video features are extracted, is to classify the activity on each clip as the videos of the ActivityNet are untrimmed and may be an activity or not (background). To perform this classification a RNN is used. More specifically a LSTM network which tries to exploit long term correlations and perform a prediction of the video sequence. This stage is the one which has been trained.

The structure of the network can be seen on the next figure.

![Network Pipeline][network-pipeline]

To reproduce all the process of the pipeline, there is a [detailed guide](https://github.com/imatge-upc/activitynet-2016-cvprw/blob/master/misc/step_by_step_guide.md) about how to reproduce all the steps with the scripts provided.

## Related work

* Tran, D., Bourdev, L., Fergus, R., Torresani, L., & Paluri, M. (2015, December). Learning spatiotemporal features with 3d convolutional networks. In 2015 IEEE International Conference on Computer Vision (ICCV) (pp. 4489-4497). IEEE. [[paper](http://arxiv.org/pdf/1412.0767.pdf)] [[code](https://github.com/facebook/C3D)]
* Sharma, S., Kiros, R., & Salakhutdinov, R. (2015). Action recognition using visual attention. arXiv preprint arXiv:1511.04119. [[paper](http://arxiv.org/pdf/1511.04119.pdf)][[code](https://github.com/kracwarlock/action-recognition-visual-attention)]
* Yeung, S., Russakovsky, O., Mori, G., & Fei-Fei, L. (2015). End-to-end Learning of Action Detection from Frame Glimpses in Videos. arXiv preprint arXiv:1511.06984. [[paper](http://arxiv.org/pdf/1511.06984.pdf)]
* Yeung, Serena, et al. "Every moment counts: Dense detailed labeling of actions in complex videos." arXiv preprint arXiv:1507.05738 (2015).[[paper](http://arxiv.org/pdf/1507.05738v2.pdf)]
* Baccouche, M., Mamalet, F., Wolf, C., Garcia, C., & Baskurt, A. (2011, November). Sequential deep learning for human action recognition. In International Workshop on Human Behavior Understanding (pp. 29-39). Springer Berlin Heidelberg. [[paper](https://www.researchgate.net/profile/Moez_Baccouche/publication/221620711_Sequential_Deep_Learning_for_Human_Action_Recognition/links/53eca3470cf250c8947cd686.pdf)]
* Shou, Zheng, Dongang Wang, and Shih-Fu Chang. "Temporal Action Localization in Untrimmed Videos via Multi-stage CNNs." [[paper](http://dvmmweb.cs.columbia.edu/files/dvmm_scnn_paper.pdf)] [[code](https://github.com/zhengshou/scnn)]

## Acknowledgements

We would like to especially thank Albert Gil Moreno and Josep Pujal from our technical support team at the Image Processing Group at the UPC.

| ![Albert Gil][image-albert] | ![Josep Pujal][image-josep]  |
| :---: | :---: |
| [Albert Gil](web-albert)  |  [Josep Pujal](web-josep) |



## Contact
If you have any general doubt about our work or code which may be of interest for other researchers, please use the [issues section](https://github.com/imatge-upc/activitynet-2016-cvprw/issues)
on this github repo. Alternatively, drop us an e-mail at [xavier.giro@upc.edu](mailto:xavier.giro@upc.edu).


<!--Images-->
[image-alberto]: misc/images/alberto_montes.jpg "Alberto Montes"
[image-amaia]: misc/images/amaia_salvador.jpg "Amaia Salvador"
[image-xavier]: misc/images/xavier_giro.jpg "Xavier Giró-i-Nieto"
[image-santi]: misc/images/santi_pascual.jpg "Santiago Pascual"
[image-albert]: misc/images/albert_gil.jpg "Albert Gil"
[image-josep]: misc/images/josep_pujal.jpg "Josep Pujal"

[image-upc-logo]: misc/images/upc_etsetb.jpg

[network-pipeline]: misc/images/network_pipeline.jpg

<!--Links-->
[web-alberto]: https://www.linkedin.com/in/albertomontesg
[web-xavier]: https://imatge.upc.edu/web/people/xavier-giro
[web-albert]: https://imatge.upc.edu/web/people/albert-gil-moreno
[web-josep]: https://imatge.upc.edu/web/people/josep-pujal
[web-amaia]: https://imatge.upc.edu/web/people/amaia-salvador
[web-santi]: https://github.com/santi-pdp

[activitynet-challenge]: http://activity-net.org/challenges/2016/
[activitynet-dataset]: http://activity-net.org/download.html
[keras]: http://keras.io/
