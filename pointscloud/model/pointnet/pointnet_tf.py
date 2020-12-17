import tensorflow as tf
import tensorflow.keras as keras
import logging as lg
import numpy as np

class pointnet(object):
    def __init__(self, input_size, output_size, learning_rate, is_training=True):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.is_training = is_training

        self.model = None
        self.optimize = keras.optimizers.Adam(self.learning_rate)
        self.log = lg.getLogger(self.__class__.__name__)
        self.log.info('parameters - is:{}, os:{}, lr:{}, is_training:{}'.format(input_size, output_size, learning_rate, is_training))
    
    def create_model(self):
        data_input, fc3 = self.forward()
        self.model = keras.Model(inputs=data_input, outputs=fc3)
        self.model.compile(optimizer=self.optimize, loss=tf.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
        self.model.summary()

    def forward(self):
        data_input = keras.Input(shape=self.input_size[1:], dtype=tf.float32)
        self.log.info('input shape:{}'.format(data_input.shape))
        
        conv1 = self.conv1_block(data_input, 64, 1)
        self.log.info('conv1 shape:{}'.format(conv1.shape))

        conv2 = self.conv1_block(conv1, 128, 1)
        self.log.info('conv2 shape:{}'.format(conv2.shape))

        conv3 = self.conv1_block(conv2, 1024, 1)
        self.log.info('conv3 shape:{}'.format(conv3.shape))
        
        # conv3 = keras.layers.Reshape(list(conv3.shape[1:]) + [1])(conv3)
        # self.log.info('reshape shape:{}'.format(conv3.shape))
        maxpool = keras.layers.MaxPool1D(self.input_size[1], strides=1)(conv3)
        self.log.info('maxpool shape:{}'.format(maxpool.shape))
        
        poolshape = maxpool.shape.as_list()[1:]
        feature = keras.layers.Reshape([np.product(poolshape)])(maxpool)
        self.log.info('feature shape:{}'.format(feature.shape))
        
        fc1 = self.fully_connected(feature, 512)
        fc2 = self.fully_connected(fc1, 256)
        fc3 = self.fully_connected(fc2, self.output_size[1], use_bn=False, use_activation=False)
        self.log.info('fc3 shape:{}'.format(fc3.shape))

        return data_input, fc3


    def conv1_block(self, x_input, channels, kernel_size, padding='valid', use_bias=True, use_bn=True):
        x = keras.layers.Conv1D(channels, kernel_size, padding=padding, use_bias=use_bias)(x_input)
        if use_bn:
            x = keras.layers.BatchNormalization()(x)
        return keras.activations.relu(x)  

    def fully_connected(self, x_input, units_num, use_bias=True, use_bn=True, use_activation=True):
        fc = keras.layers.Dense(units_num, 'relu', use_bias=use_bias)(x_input)
        if use_bn:
            fc = keras.layers.BatchNormalization()(fc)
        if use_activation:
            fc = keras.activations.relu(fc)
        return fc

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load_weights(path)

