import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


class pointnet2(object):
    def __init__(self, input_size, output_size, is_trainging=True):
        self.input_size = input_size
        self.output_size = output_size
        self.is_training = is_trainging

    def create_model(self):
        data_input = keras.layers.Input(shape=self.input_size, dtype=tf.float32)


    def msg_layer(self):
        pass

    def mrg_layer(self):
        pass


class FPS(object):
    def __init__(self, kernel_points_num, input_size):
        self.kernel_points_num= kernel_points_num
        self.input_size = input_size
        self.batch_size = input_size[0]
        self.points_num = input_size[1]
        self.feature_dim = input_size[2]

    def fps_layer_gpu(self, inputs, fps_name):
        
        
        

    def fps_layer_tensor(self, inputs):
        long_distences = tf.zeros([self.batch_size, self.kernel_points_num])
        random_index = tf.constant(np.random.randint(0, self.points_num, self.batch_size))
        for 

