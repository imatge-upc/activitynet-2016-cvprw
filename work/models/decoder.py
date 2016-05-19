""" Decoder models for activity recognition
This models are proposals as part of my thesis with the aim to
recognise ativities.
This decoder models go after an encoder which using the c3d model,
extract the features of video clips at the fc6 layer.
So the input for the decoder models are 4096 features.
"""
from keras.layers import BatchNormalization, Dense, Input, Masking, Merge, TimeDistributed, merge
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential


def RecurrentActivityClassificationNetwork(batch_size, timesteps, summary=False, stateful=True):
    input_features = Input(batch_shape=(batch_size, timesteps, 4096,), name='features')
    input_normalized = BatchNormalization()(input_features)
    lstm1 = LSTM(512, return_sequences=True, stateful=stateful, name='lstm1')(input_normalized)
    lstm2 = LSTM(512, return_sequences=True, stateful=stateful, name='lstm2')(lstm1)
    output = TimeDistributed(Dense(201, activation='softmax'), name='fc')(lstm2)

    model = Model(input=input_features, output=output)

    if summary:
        print(model.summary())
    return model


def RecurrentFeedbackActivityDetectionNetwork(batch_size, timesteps, summary=False, stateful=True):
    input_features = Input(batch_shape=(batch_size, timesteps, 4096,), name='features')
    input_normalized = BatchNormalization()(input_features)
    previous_output = Input(batch_shape=(batch_size, timesteps, 202,), name='prev_output')
    merging = merge([input_normalized, previous_output], mode='concat', concat_axis=-1)
    lstm1 = LSTM(512, return_sequences=True, stateful=True, name='lstm1')(merging)
    lstm2 = LSTM(512, return_sequences=True, stateful=True, name='lstm2')(lstm1)
    output = TimeDistributed(Dense(201, activation='softmax'), name='fc')(lstm2)
    model = Model(input=[input_features, previous_output], output=output)

    if summary:
        print(model.summary())
    return model

def RecurrentRegressionModel(batch_size, timesteps, summary=False):
    input_features = Input(batch_shape=(batch_size, timesteps, 4096,), name='features')
    input_masked = Masking(mask_value=0.)(input_features)
    lstm1 = LSTM(512, return_sequences=True, name='lstm1')(input_masked)
    lstm2 = LSTM(512, return_sequences=True, name='lstm2')(lstm1)

    temporal_prediction = Dense(10, activation='sigmoid')(lstm2)
    flag = Dense(5, activation='softmax')(lstm2)
    class_prediction = Dense(201, activation='softmax')(lstm2)

    model = Model(input=input_features, output=[temporal_prediction, flag, class_prediction])

    if summary:
        print(model.summary())
    return model

def BidirectionalLSTMModel(hidden_units, nb_classes, summary=False):
    def fork (model, n=2):
        forks = []
        for i in range(n):
            f = Sequential()
            f.add (model)
            forks.append(f)
        return forks

    left = Sequential()
    left.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
                   forget_bias_init='one', return_sequences=True, activation='tanh',
                   inner_activation='sigmoid', input_shape=(4096,)))
    right = Sequential()
    right.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
                   forget_bias_init='one', return_sequences=True, activation='tanh',
                   inner_activation='sigmoid', input_shape=(4096,), go_backwards=True))

    model = Sequential()
    model.add(Merge([left, right], mode='sum'))

    # Add second Bidirectional LSTM layer

    left, right = fork(model)

    left.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
                   forget_bias_init='one', return_sequences=True, activation='tanh',
                   inner_activation='sigmoid'))

    right.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
                   forget_bias_init='one', return_sequences=True, activation='tanh',
                   inner_activation='sigmoid',  go_backwards=True))

    #Rest of the stuff as it is

    model = Sequential()
    model.add(Merge([left, right], mode='sum'))

    model.add(TimeDistributed(Dense(nb_classes, activation='softmax')))

    if summary:
        print(model.summary())

    return model
