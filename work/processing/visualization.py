import matplotlib.pyplot as plt
import numpy as np


def plot_sequence(sequence, fps=1):
    assert len(sequence.shape) == 1

    im = sequence.reshape(1, sequence.shape[0]).astype(np.dtype32)/255
    plt.imshow(im)
    pass
