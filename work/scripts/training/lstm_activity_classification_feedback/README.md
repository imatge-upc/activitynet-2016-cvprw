# Activity Classification

A network using a two LSTM layers with 512 neurons each one and a fc layer at the end with 201
outputs, one for each class to classify (200) plus the background.
The input is the 4096 features extracted for every 16-frames clip of each video. This features
vector is concatenated with a one-hot representation of the previous output. This is trying to
achieve a more continuous and soft output.

Experiments done:
* **#1**: default

## Results
### Experiment 1

```

```
