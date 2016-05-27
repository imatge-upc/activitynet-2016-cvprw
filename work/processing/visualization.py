import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_sequence(sequence, fps=1, nb_classes=201, probability=False, title=''):
    assert len(sequence.shape) == 1
    nb_instances = sequence.shape[0]
    v_max = 1 if probability else nb_classes
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=v_max)

    x = np.arange(nb_classes)*16/fps
    y = np.arange(2)

    plt.figure(num=None, figsize=(18, 1), dpi=100)
    plt.contourf(x, y, np.broadcast_to(sequence, (2, nb_instances)), norm=normalize, interpolation='nearest')
    plt.title(title)
    plt.show()
