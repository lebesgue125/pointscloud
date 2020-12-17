import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from model.pointnet.tf_op.tf_lib_ops import lib_center


class FarthestPointSample(object):
    def __init__(self, op_lib):
        self.farthest_op = op_lib.get_lib_ops('tf_op_cu')

    def farthest_point_sample(self, npoint, inp):
        if self.farthest_op:    
            return self.farthest_op.farthest_point_sample(inp, npoint)    
        else:
            raise Exception('tf_op_cu lib not exiet!')

fps = FarthestPointSample(lib_center)

@tf.RegisterGradient('FarthestPointSample')
def _farthest_point_sample_grad(op, grad):
    idx=op.outputs[0]
    return fps.farthest_op.farthest_point_sample_grad(idx, grad)
