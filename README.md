# Temporal Activity Detection in Untrimmed Videos with Recurrent Neural Networks

This is the project page of the UPC team participating in the [ActivityNet Challenge][activitynet-challenge] for CVPR 2016.

| ![Alberto Montes][image-alberto] | ![Amaia Salvador](imatge-amaia) | ![Xavier Giró-i-Nieto][image-xavier] | ![Santiago Pascual][image-santi] |
| :---: | :---: | :---: | :---: |
| Main contributor | Advisor | Advisor | Co-advisor |
| Issey Masuda Mora | [Amaia Salvador](web-amaia) | [Xavier Giró-i-Nieto][web-xavier] | Santiago Pascual |

Institution: [Universitat Politècnica de Catalunya](http://www.upc.edu).

![Universitat Politècnica de Catalunya][image-upc-logo]


## Abstract

Deep learning techniques have been proven to be a great success for tasks like object detection and classification.
They have achieve huge accuracy on images but on videos where the temporal dimension is present, more new techniques are required to face task over them.

Activity classification and temporal activity location require new models which try to explode the temporal correlations the videos present to achieve good results on this tasks. The work presented try to face this tasks, for both activity classification and temporal activity localization using the [ActivityNet Dataset][activitynet-dataset].

This work propose to face the tasks with a two stage pipeline. The first stage is to extract video features from the C3D which exploit temporal correlations and then a RNN made up by LSTM cells which try to learn long-term correlations and returning a sequence of activities along the video that will help to classify and temporally localize activities.


## Dependencies

This project is build using the [Keras](https://github.com/fchollet/keras) library for Deep Learning, which can use as a backend both [Theano](https://github.com/Theano/Theano)
and [TensorFlow](https://github.com/tensorflow/tensorflow).

We have used Theano in order to develop the project because it supported 3D convolutions and pooling required to run the C3D network.

For a further and more complete of all the dependencies used within this project, check out the requirements.txt provided within the project. This file will help you to recreate the exact same Python environment that we worked with.



## Acknowledgements

We would like to especially thank Albert Gil Moreno and Josep Pujal from our technical support team at the Image Processing Group at the UPC.

| ![Albert Gil][image-albert] | ![Josep Pujal][image-josep]  |
| :---: | :---: |
| [Albert Gil](web-albert)  |  [Josep Pujal](web-josep) |



## Contact
If you have any general doubt about our work or code which may be of interest for other researchers, please use the [issues section](https://github.com/imatge-upc/activitynet-2016-cvprw/issues)
on this github repo. Alternatively, drop us an e-mail at [xavier.giro@upc.edu](mailto:xavier.giro@upc.edu).


<!--Images-->
[image-alberto]: https://raw.githubusercontent.com/imatge-upc/activitynet-2016-cvprw/master/misc/images/alberto_montes.jpg "Alberto Montes"
[image-amaia]: https://raw.githubusercontent.com/imatge-upc/activitynet-2016-cvprw/master/misc/images/xavier_giro.jpg Amaia Salvador"
[image-xavier]: https://raw.githubusercontent.com/imatge-upc/activitynet-2016-cvprw/master/misc/images/xavier_giro.jpg "Xavier Giró-i-Nieto"
[image-santi]: https://raw.githubusercontent.com/imatge-upc/activitynet-2016-cvprw/master/misc/images/santi_pascual.jpg "Santiago Pascual"
[image-albert]: https://raw.githubusercontent.com/imatge-upc/activitynet-2016-cvprw/master/misc/images/albert_gil.jpg "Albert Gil"
[image-josep]: https://raw.githubusercontent.com/imatge-upc/activitynet-2016-cvprw/master/misc/images/josep_pujal.jpg "Josep Pujal"

[image-model]: https://raw.githubusercontent.com/imatge-upc/activitynet-2016-cvprw/master/misc/images/model.jpg
[image-upc-logo]: https://raw.githubusercontent.com/imatge-upc/activitynet-2016-cvprw/master/misc/images/upc_etsetb.jpg

<!--Links-->
[web-xavier]: https://imatge.upc.edu/web/people/xavier-giro
[web-albert]: https://imatge.upc.edu/web/people/albert-gil-moreno
[web-josep]: https://imatge.upc.edu/web/people/josep-pujal
[web-amaia]: https://imatge.upc.edu/web/people/amaia-salvador

[activitynet-challenge]: http://activity-net.org/challenges/2016/
[activitynet-dataset]: http://activity-net.org/download.html
[keras]: http://keras.io/
