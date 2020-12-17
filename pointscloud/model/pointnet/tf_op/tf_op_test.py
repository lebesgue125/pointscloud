import tensorflow as tf
import numpy as np 
import tf_lib_ops as tlo 
import logging as lg
import os, sys
import time
BASE_DIR = os.path.dirname('/home/lebesgue/PointCloud/3DProject/')
sys.path.append(BASE_DIR)
from dataloader.ModelNetLoader import modelnetload

def farthest_point_sample(npoint, inp):
    if tlo.ops_lib_exist('tf_op_cu'):
        op = tlo.get_lib_ops('tf_op_cu')
        return op.farthest_point_sample(inp, npoint)    
    else:
        raise Exception('tf_op_cu lib not exiet!')


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    lib_path = '/home/lebesgue/PointCloud/3DProject/model/lib'
    LOG_FORMAT = "%(asctime)s-%(levelname)s-[%(name)s]:%(message)s"
    lg.basicConfig(filename='data_test.log', level=lg.INFO, format=LOG_FORMAT)  
    tlo.load_tf_lib(lib_path)
    tlo.print_loaded_lib()

    root = '/home/lebesgue/data/Kitti/modelnet40_normal_resampled'
    resource_file = 'modelnet10_train.txt'
    category_file = 'modelnet10_shape_names.txt'
    test_file = 'modelnet10_train.txt' 

    batch_size = 128
    max_points = 1500
    dimension = 3
    out_points = 512
    loader = modelnetload(root, resource_file, category_file, max_points=max_points, 
                        dimension=dimension, data_augmentation=True)
    data, _ = loader.iter(batch_size)
    index, points = farthest_point_sample(out_points, data)
    print(index[0], points[0])
    for i in range(100):
        index_p = index[0, i].numpy()
        point = points[0,i].numpy()
        print(index_p, point, data[0,index_p], point-data[0,index_p])