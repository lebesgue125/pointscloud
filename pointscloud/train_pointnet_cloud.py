import train_pointnet as tp
import logging as lg
import os


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    LOG_FORMAT = "%(asctime)s-%(levelname)s-[%(name)s]:%(message)s"
    lg.basicConfig(filename='data_test.log', level=lg.INFO, format=LOG_FORMAT)    

    root = '/aul/homes/ywu048/pointscloud/modelnet40_normal_resampled'
    resource_file = 'modelnet10_train.txt'
    category_file = 'modelnet10_shape_names.txt'  
    test_file = 'modelnet10_train.txt'  
    tp.train_pointnet(root, resource_file, test_file,category_file)