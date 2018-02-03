from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import TimeDistributed
from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras import backend as K


def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length//16

    return get_output_length(width), get_output_length(height)
    
def nn_base(input_tensor=(None, None, 3), trainable=False):
    # Determine proper input shape
    input_shape = (None, None, 3)
    
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
            print('blablablab')
        else:
            img_input = input_tensor