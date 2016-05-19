# Activity Classification

A network using a two LSTM layers with 512 neurons each one and a fc layer at the end with 201
outputs, one for each class to classify (200) plus the background.
The Input is the 4096 features extracted for each 16-frames clip of every video.

Experiments done:
* **#1**: as is described
* **#2**: adding a normalization layer at the input of the features

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

```
