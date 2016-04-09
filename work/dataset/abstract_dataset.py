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
    def __init__(self, instances=None):
        self.instances=None
        pass

    def get_subset_instances(self, subset):
        """ Returns all the instances of the dataset that belongs to the same
        subset (example: 'train' or 'validation')
        Args
            * subset: (string or list): name or list of names of the subset
                instances you want to get.
        """
        if isinstance(subset, str):
            return [i for i in self.instances if i.subset == subset]
        elif isinstance(subset, list):
            return [i for i in self.instances if i.subset in subset]


class AbstractInstance(object):
    """ Class that represents an instance of a dataset. This instance is not
    loaded, but gives the path where the information is stored.
    At the path is where the input is stored (this is due to high volume dataset difficult to have them loaded in memory).
    The output could be the category or the value at the output.
    """
    def __init__(self, type_, path, output=None):
        assert isinstance(type_, InstanceType)
        self.type_ = type_
        self.path = path
        self.output = output

    def get_instance_values(self, *args, **kwargs):
        """ Method to retrieve the input values of the instance stored at the
        given path.
        """
        if type_ is InstanceType.IMAGE:
            image = load_img(self.path, **kwargs)
            return img_to_array(image)
        elif type_ is InstanceType.VIDEO:
            return video_to_array(self.path, **kwargs)
        elif type_ is InstanceType.IMAGE_3D:
            # TODO: there is no specific way to store it
            pass
