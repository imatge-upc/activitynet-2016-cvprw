import keras.backend as K
from keras.engine.topology import InputSpec, Layer


class AttentionFilters(Layer):

    def __init__(self, num_filters, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.num_filters = num_filters

    def build(self, input_shape):
        value_shape, features_shape = input_shape[0], input_shape[1]

        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        input_dim = input_shape[1]
        initial_weight_value = np.random.random((input_dim, output_dim))
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return K.dot(x, self.W)

    def get_output_shape_for(self, input_shape):
        return (self.num_filters, input_shape[-1])
