import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_sequence(sequence, fps=1, nb_classes=201, title=''):
    assert len(sequence.shape) == 1
    nb_instances = sequence.shape[0]
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=nb_classes)

    plt.figure(num=None, figsize=(18, 1), dpi=100)
    plt.contourf(np.broadcast_to(sequence, (2, nb_instances)), norm=normalize)
    plt.title(title)
    plt.show()
