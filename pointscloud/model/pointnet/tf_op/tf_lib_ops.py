import tensorflow as tf
import os
import logging as lg


class LibCenter(object):
    def __init__(self, paths = None):
        self.ops_mapping = {}
        self.paths = paths
        if paths:
            for path in paths:
                self.load_tf_lib(path)
          
    def load_tf_lib(self, path):
        if path is not None and os.path.exists(path):
            if os.path.isdir(path):
                for file_name in os.listdir(path):
                    self.load_tf_lib(os.path.join(path, file_name))
            else:
                file_name = os.path.basename(path)
                if file_name.endswith('.so'):
                    tf_module = tf.load_op_library(path)
                    self.ops_mapping[file_name.split('.')[0]] = tf_module
                    lg.info('successful loaded lib: {}, from: {}'.format(file_name, path))
        else:
            lg.info('Path is not exist! {}'.format(path))

    def ops_lib_exist(self, name):
        return name in self.ops_mapping


    def get_lib_ops(self, name):
        return self.ops_mapping.get(name, None)

    def print_loaded_lib(self):
        lg.info('.'.join(self.ops_mapping.keys()))

path = ['/home/lebesgue/PointCloud/3DProject/model/lib']
lib_center = LibCenter(path)