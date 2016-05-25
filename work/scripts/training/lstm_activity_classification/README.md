# Activity Classification

A network using a two LSTM layers with 512 neurons each one and a fc layer at the end with 201
outputs, one for each class to classify (200) plus the background.
The Input is the 4096 features extracted for each 16-frames clip of every video.

Experiments done:
* **#1**: as is described
* **#2**: adding a normalization layer at the input of the features
* **#3**: Changing the network keeping the normalization and putting 3 LSTM layers with 1024
* **#4**: Seeing that the training the validation loss increassed a lot and teh val accuracy didn't
reduced, now the learning rate has been decreassed to 0.0001 and added a 0.5 dropout at the first
LSTM.
neureons each

## Results
### Experiment 1

```
[RESULTS] Performance on ActivityNet untrimmed video classification task.
	Mean Average Precision: 0.507605336599
	Hit@3: 0.694352844188
	Avg Hit@3: 0.694249793899
```

### Experiment 2

```
[RESULTS] Performance on ActivityNet untrimmed video classification task.
	Mean Average Precision: 0.471807446716
	Hit@3: 0.656977942692
	Avg Hit@3: 0.656565656566
```

### Experiment 3

```

```
