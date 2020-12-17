import tensorflow as tf
from model.pointnet.pointnet_tf import pointnet
from dataloader.ModelNetLoader import modelnetload
import util.dataprocesses as dp
import numpy as np
import logging as lg
import time
import os

def train_pointnet(root, resource_file, test_file, category_file, save_path):   
    batch_size = 32
    max_points = 2500
    dimension = 3
    learning_rate = 1e-4
    epoch = 10000

    loader = modelnetload(root, resource_file, category_file, max_points=max_points, 
                          dimension=dimension, data_augmentation=True)
   
    test_loader = modelnetload(root, test_file, category_file)
    input_size = (batch_size, max_points, dimension)
    output_size = (batch_size, loader.cat_num)
    model = pointnet(input_size, output_size, learning_rate)
    model.create_model()
    
    save_threshold = 0.8

    for i in range(epoch):
        data, cls = loader.iter(128)
        lg.debug('input data shape {}'.format(data.shape))
        lg.debug('data: {}'.format(data))
        onehot_cls = dp.generate_onehot(cls, loader.cat_num)
        
        history = model.model.fit(x=data, y=onehot_cls, batch_size=batch_size, epochs=10, 
                                  workers=5, use_multiprocessing=True)


        lg.info('average loss: {} at {} iter'.format(np.mean(history.history['loss']), i))
        lg.info('average accuracy: {} at {} iter'.format(np.mean(history.history['accuracy']), i))

        if i % 10 == 0:
            test_x, test_y = test_loader.iter(250)
            test_y = dp.generate_onehot(test_y, loader.cat_num)
            result = model.model.evaluate(test_x, test_y, batch_size=128)
            lg.info('evaluation - loss: {}, accuracy: {}'.format(result[0],result[1]))
            if result[1] > save_threshold:
                model.save(save_path)
                
    

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    LOG_FORMAT = "%(asctime)s-%(levelname)s-[%(name)s]:%(message)s"
    lg.basicConfig(filename='data_test.log', level=lg.INFO, format=LOG_FORMAT)  

    root = '/home/lebesgue/data/Kitti/modelnet40_normal_resampled'
    resource_file = 'modelnet10_train.txt'
    category_file = 'modelnet10_shape_names.txt'
    test_file = 'modelnet10_train.txt'
    save_path = 'trainedModel'
    train_pointnet(root, resource_file, test_file, category_file, save_path)