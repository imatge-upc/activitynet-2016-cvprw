import random
from enum import Enum

from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.video import video_to_array


class InstanceType(Enum):
    IMAGE = 0           # Image data
    VIDEO = 1           # Video data
    IMAGE_3D = 2        # 3D Images (used in medical applications for example)

class AbstractDataset(object):
    """ Class that represent a dataset which contains lots of instances
    This dataset can return all the instances from the desired subset
    (train, validation, test)
    """
    def __init__(self):
        pass

    def get_instances(self, random_sorted=False):
        if random_sorted:
            random.shuffle(self.instances)
        return self.instances

    # def get_subset_instances(self, subset, random_sorted=False):
    #     """ Returns all the instances of the dataset that belongs to the same
    #     subset (example: 'train' or 'validation')
    #     Args
    #         * subset: (string or list): name or list of names of the subset
    #             instances you want to get.
    #     """
    #     if isinstance(subset, str):
    #         instances = [i for i in self.instances if i.subset == subset]
    #     elif isinstance(subset, list):
    #         instances = [i for i in self.instances if i.subset in subset]
    #     if random_sorted:
    #         random.shuffle(instances)
    #     return instances

    def import_instances():
        """ Method to override to import the instances of the dataset. It could
        be from a file or whatever you want.
        """
        pass

class AbstractInstance(object):
    """ Class that represents an instance of a dataset. This instance is not
    loaded, but gives the path where the information is stored.
    At the path is where the input is stored (this is due to high volume dataset
    difficult to have them loaded in memory).
    The output could be the category or the value at the output.
    """
    def __init__(self, instance_id, output):
        self.instance_id = instance_id
        self.output = output
