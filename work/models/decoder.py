""" Decoder models for activity recognition
This models are proposals as part of my thesis with the aim to
recognise ativities.
This decoder models go after an encoder which using the c3d model,
extract the features of video clips at the fc6 layer.
So the input for the decoder models are 4096 features.
"""
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential


def RecurrentNetwork(batch_size, sequence_length, summary=False, stateful=True):
    model = Sequential()
    model.add(LSTM(512, batch_input_shape=(batch_size, sequence_length, 4096),
        return_sequences=True, stateful=stateful, name='lstm1'))
    model.add(LSTM(512, batch_input_shape=(batch_size, sequence_length, 4096),
        return_sequences=True, stateful=stateful, name='lstm2'))
    model.add(TimeDistributed(Dense(201, activation='softmax'), name='fc-o'))

    if summary:
        print(model.summary())
    return model

def Classifier(summary=False):
    model = Sequential()
    model.add(Dense(4096, input_shape=(4096,), activation='relu', name='fc7'))
    model.add(Dense(201, activation='softmax', name='fc8'))

    if summary:
        print(model.summary())
    return model

def ParallelClassifier(summary=False):
    model_yn, model_cl = Sequential(), Sequential()
    # Model to obtain whether an activity is happening or not
    model_yn.add(Dense(2048, input_shape=(4096,), activation='relu', name='yn_fc7'))
    model_yn.add(Dense(2, activation='softmax', name='yn_fc8'))

    # Model to classify the activity only in the case an activity
    model_cl.add(Dense(4096, input_shape=(4096,), activation='relu', name='fc7'))
    model_cl.add(Dense(200, activation='softmax', name='fc8'))

    if summary:
        print('Model to detect if an activity is happening:')
        print(model_yn.summary())
        print('Model to obtain which activity is happening:')
        print(model_cl.summary())
    return model_yn, model_cl
