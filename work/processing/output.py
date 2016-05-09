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

def get_temporal_predictions_2(x, fps=1, clip_length=16, k=3):
    results = []
    predictions = []

    new_prediction = []
    pointer = 0.
    for clip in x:
        if clip != 0 and not new_prediction:
            new_prediction = [int(pointer)]
        elif new_prediction and clip == 0:
            new_prediction.append(int(pointer))
            predictions.append(new_prediction.copy())
            new_prediction = []
        pointer += 1.

    for prediction in predictions:
        segment = x[prediction[0]:prediction[1]]
        top_k, scores = get_top_k_predictions_score(segment, k=k)
        for activity_index in range(len(top_k)):
            if scores[activity_index] > 0:
                results.append({
                    'score': scores[activity_index],
                    'segment': [i * 16 / fps for i in prediction],
                    'label': top_k[activity_index]
                })

    return results

def get_temporal_predictions_3(x, fps=1, clip_length=16, k=1):
    results = []
    predictions = []

    start_activity = 0
    end_activity = 0
    pointer = 0.
    previous_activity = 0
    for clip in x:
        if clip != 0 and start_activity == 0:
            start_activity = pointer
        elif previous_activity != 0 and clip == 0:
            end_activity = pointer
        pointer += 1.
        previous_activity = clip

    top_activity = get_top_k_predictions(x, k=1)
    if top_activity:
        results = [{
            'score': 1,
            'segment': [
                start_activity * clip_length / fps,
                end_activity * clip_length / fps
            ],
            'label': top_activity[0]
        }]

    return results
