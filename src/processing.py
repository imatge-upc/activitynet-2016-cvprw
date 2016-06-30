import numpy as np


def get_class_prediction(sequence_class_prob, k=3):
    ''' From predicted classes for a sequence of clips corresponfing to a video, it
    returns the top k classes and its respective scores
    '''
    class_prob = np.mean(sequence_class_prob, mean=0)
    labels_index = np.argsort(class_prob[1:])[::-1] + 1
    scores = sequence_class_prob[labels_index]
    return labels_index[:k], scores[:k]

def smoothing(x, k=5):
    ''' Applies a mean filter to an input sequence. The k value specifies the window
    size. window size = 2*k
    '''
    l = len(x)
    s = np.arange(-k, l-k)
    e = np.arange(k, l+k)
    s[s<0] = 0
    e[e>=l] = l-1
    y = np.zeros(x.shape)
    for i in range(l):
        y[i] = np.mean(x[s[i]:e[i]], axis=0)
    return y
