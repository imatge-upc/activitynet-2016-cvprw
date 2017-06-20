import numpy as np


def video_to_array(video_path,
                   resize=None,
                   start_frame=0,
                   end_frame=None,
                   length=None,
                   dim_ordering='th'):
    ''' Convert the video at the path given in to an array
    Args:
        video_path (string): path where the video is stored
        resize (Optional[tupple(int)]): desired size for the output video.
            Dimensions are: height, width
        start_frame (Optional[int]): Number of the frame to start to read
            the video
        end_frame (Optional[int]): Number of the frame to end reading the
            video.
        length (Optional[int]): Number of frames of length you want to read
            the video from the start_frame. This override the end_frame
            given before.
    Returns:
        video (nparray): Array with all the data corresponding to the video
                         given. Order of dimensions are: channels, length
                         (temporal), height, width.
    Raises:
        Exception: If the video could not be opened
    '''
    import cv2
    if cv2.__version__ >= '3.0.0':
        CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
        CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
    else:
        CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
        CAP_PROP_POS_FRAMES = cv2.cv.CV_CAP_PROP_POS_FRAMES

    if dim_ordering not in ('th', 'tf'):
        raise Exception('Invalid dim_ordering')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception('Could not open the video')

    num_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
    if start_frame >= num_frames or start_frame < 0:
        raise Exception('Invalid initial frame given')
    # Set up the initial frame to start reading
    cap.set(CAP_PROP_POS_FRAMES, start_frame)
    # Set up until which frame to read
    if end_frame:
        end_frame = end_frame if end_frame < num_frames else num_frames
    elif length:
        end_frame = start_frame + length
        end_frame = end_frame if end_frame < num_frames else num_frames
    else:
        end_frame = num_frames
    if end_frame < start_frame:
        raise Exception('Invalid ending position')

    frames = []
    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            return None

        if resize:
            # The resize of CV2 requires pass firts width and then height
            frame = cv2.resize(frame, (resize[1], resize[0]))
        frames.append(frame)

    video = np.array(frames, dtype=np.float32)
    if dim_ordering == 'th':
        video = video.transpose(3, 0, 1, 2)
    return video


def get_num_frames(video_path):
    ''' Return the number of frames of the video track of the video given '''
    import cv2
    if cv2.__version__ >= '3.0.0':
        CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
    else:
        CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception('Could not open the video')
    num_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
    return num_frames


def get_duration(video_path):
    ''' Return the duration of the video track of the video given '''
    import cv2
    if cv2.__version__ >= '3.0.0':
        CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
        CAP_PROP_FPS = cv2.CAP_PROP_FPS
    else:
        CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
        CAP_PROP_FPS = cv2.cv.CV_CAP_PROP_FPS

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception('Could not open the video')
    num_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(CAP_PROP_FPS))
    duration = num_frames / fps
    return duration
