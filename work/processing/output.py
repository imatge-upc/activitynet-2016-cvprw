""" Utils to process the output data
"""
import numpy as np


# To utils to process the output data...


def sequence_predicted_class(x):
    """ Expect to be th input a numpy array
    """
    return np.argmax(np.bincount(x)[1:])+1

def get_top_k_predictions(x, k=3):
    return np.argsort(np.bincount(x)[1:])[::-1][:k] + 1

def get_top_k_predictions_score(x, k=3):
    counts = np.bincount(x)
    top_k = np.argsort(counts[1:])[::-1][:k] + 1
    scores = counts[top_k]/np.sum(counts[1:])
    return top_k, scores

def get_video_ground_truth(video):
    if not video.instances:
        raise Exception('The video needs to have it instances generated')

    return np.array([ins.output for ins in video.instance])

def get_temporal_predictions(x, fps=1, clip_length=16):
    predictions = []
    counts = np.bincount(x)
    if len(counts) == 1:
        return predictions
    predicted_class = np.argmax(counts[1:]) + 1

    new_prediction = {}
    pointer = 0.
    for clip in x:
        if clip == predicted_class and not new_prediction:
            new_prediction = {
                'score': 1,
                'segment': [pointer / fps],
                'label': predicted_class
            }
        elif new_prediction and clip != predicted_class:
            new_prediction['segment'].append(pointer / fps)
            predictions.append(new_prediction.copy())
            new_prediction = {}
        pointer += clip_length

    return predictions
