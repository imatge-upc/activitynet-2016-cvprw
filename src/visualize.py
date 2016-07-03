import re

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, Normalize

%matplotlib inline

def plot_on_training(txt, max_epochs=100, max_loss=5):
    loss = re.findall('- loss: \d+\.\d+', txt)
    accuracy = re.findall('- acc: \d+\.\d+', txt)
    val_loss = re.findall('- val_loss: \d+\.\d+', txt)
    val_accuracy = re.findall('- val_acc: \d+\.\d+', txt)

    loss = np.array([float(x[8:]) for x in loss], dtype=np.float32)
    accuracy = np.array([float(x[7:]) for x in accuracy], dtype=np.float32)
    val_loss = np.array([float(x[12:]) for x in val_loss], dtype=np.float32)
    val_accuracy = np.array([float(x[11:]) for x in val_accuracy], dtype=np.float32)

    t = np.arange(1, len(val_loss)+1).astype(np.float32)
    fig, ax1 = plt.subplots(figsize=(18, 10), dpi=100)
    ax1.plot(t, loss[392::393], 'b-')
    ax1.plot(t, val_loss, 'b-.')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='b')
    ax1.set_xlim([0,max_epochs])
    ax1.set_ylim([0,max_loss])
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    ax2 = ax1.twinx()
    ax2.plot(t, accuracy[392::393], 'r-')
    ax2.plot(t, val_accuracy, 'r-.')
    ax2.set_ylabel('Accuracy', color='r')
    ax2.set_xlim([0,max_epochs])
    ax2.set_ylim([0, 1])
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    plt.show()


def visualize_temporal_activities(class_predictions, max_value=200, fps=1, title=None, legend=False):
    normalize = Normalize(vmin=1, vmax=max_value)
    normalize.clip=False
    cmap = plt.cm.Reds
    cmap.set_under('w')
    nb_instances = len(class_predictions)
    plt.figure(num=None, figsize=(18, 1), dpi=100)
    to_plot = class_predictions.astype(np.float32)
    to_plot[class_predictions==0.] = np.ma.masked
    plt.imshow(np.broadcast_to(to_plot, (20, nb_instances)), norm=normalize, interpolation='nearest', aspect='auto', cmap=cmap)
    if title:
        plt.title(title)
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)

    if legend:
        index = np.arange(0,200)
        colors_index = np.unique(to_plot).astype(np.int64)
        if 0 in colors_index:
            colors_index = np.delete(colors_index, 0)
        patches = []
        for c in colors_index:
            patches.append(mpatches.Patch(color=cmap(normalize(c)), label=dataset.labels[c][1]))
        if patches:
            plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -.2), ncol=len(patches), fancybox=True, shadow=True)
    plt.show()

def compare_temporal_activities(ground_truth, class_predictions, max_value=200, fps=1, title=None, legend=False, save_file='./img/activity_detection_sample_{}.png'):
    global count
    normalize = Normalize(vmin=1, vmax=max_value)
    normalize.clip=False
    cmap = plt.cm.Reds
    cmap.set_under('w')
    nb_instances = len(class_predictions)
    to_plot = np.zeros((20, nb_instances))
    to_plot[:10,:] = np.broadcast_to(ground_truth, (10, nb_instances))
    to_plot[10:,:] = np.broadcast_to(class_predictions, (10, nb_instances))
    to_plot = to_plot.astype(np.float32)
    to_plot[to_plot==0.] = np.ma.masked

    # Normalize the values and give them the largest distance possible between them
    unique_values = np.unique(to_plot).astype(np.int64)
    if 0 in unique_values:
        unique_values = np.delete(unique_values, 0)
    nb_different_values = len(unique_values)
    color_values = np.linspace(40, 190, nb_different_values)
    for i in range(nb_different_values):
        to_plot[to_plot == unique_values[i]] = color_values[i]

    plt.figure(num=None, figsize=(18, 1), dpi=100)
    plt.imshow(to_plot, norm=normalize, interpolation='nearest', aspect='auto', cmap=cmap)
    #plt.grid(True)
    plt.axhline(9, linestyle='-', color='k')
    plt.xlim([0,nb_instances])
    if title:
        plt.title(title)
    ax = plt.gca()
    #ax.get_yaxis().set_visible(False)
    ax.xaxis.grid(True, which='major')
    labels=['Ground\nTruth', 'Prediction']
    plt.yticks([5,15], labels, rotation="horizontal", size=13)
    plt.xlabel('Time (s)', horizontalalignment='left', fontsize=13)
    ax.xaxis.set_label_coords(0, -0.3)

    if legend:
        patches = []
        for c, l in zip(color_values, unique_values):
            patches.append(mpatches.Patch(color=cmap(normalize(c)), label=dataset.labels[l][1]))
        if patches:
            plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=len(patches), fancybox=True, shadow=True)
    #plt.show()
    plt.savefig(save_file.format(count), bbox_inches='tight')
    count += 1
