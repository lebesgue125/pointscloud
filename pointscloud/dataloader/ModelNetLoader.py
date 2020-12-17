import dataloader.DataLoader as dl
import os
import logging as lg
import numpy as np


class modelnetload(dl.dataloader):

    def __init__(self, root, resource_file, category_file, data_augmentation=False, dimension=3, max_points=2500):
        self.files_set = []
        self.cat = {}
        self.root = root
        self.max_points = max_points
        self.dimension = dimension
        self.data_augmentation = data_augmentation
        self.index = 0
        self.log = lg.getLogger(self.__class__.__name__)
        self.log.info('root: {}, resource_file: {}, category_file: {}, max_points: {}, data_augmentation: {}'.format(root, resource_file, category_file, max_points, data_augmentation))

        with open(os.path.join(root, category_file), mode='r', encoding='utf8') as cf:
            for c in cf.readlines():
                self.cat[c.strip()] = len(self.cat)
        with open(os.path.join(root, resource_file), mode='r', encoding='utf8') as rf:
            for f in rf.readlines():
                self.files_set.append(f.strip())
        
        self.files_num = len(self.files_set)
        self.cat_num = len(self.cat)

        self.log.info('Total files number is {}'.format(self.files_num))
        self.log.info('Total files number is {}'.format(self.cat_num))

        assert self.files_num > 0, 'the list of data files is empty!'
        assert self.cat_num > 0, 'the categroy file is empty!'
        
    def __getiterdata__(self):
        file_name = self.files_set[self.index % self.files_num]
        file_cat = '_'.join(file_name.split('_')[0:-1])
        cat = self.cat[file_cat]
        file_path = os.path.join(self.root, file_cat, '{}.txt'.format(file_name))
        self.log.debug('file name:{}, cat:{}, path:{}'.format(file_name, cat, file_path))

        points = np.loadtxt(file_path, delimiter=',', dtype=np.float)
        choice = np.random.choice(range(points.shape[0]), self.max_points, replace=True)
        point_set = points[choice, 0:self.dimension]

        self.log.debug('input data shape:{}'.format(points.shape))
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            self.log.debug('rotation angle: {}'.format(theta))
            self.log.debug('rotation matrix: {}'.format(rotation_matrix))
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter


        self.index += 1 
        return point_set, cat

    def iter(self, batch_size, takesample=False):
        bpoints = []
        bcat = [] 
        if takesample:
            random_set = np.random.randint(0, self.files_num, batch_size)
        for i in range(batch_size):
            if takesample:
                self.index = random_set[i]
            point_set, cat = self.__getiterdata__()
            bpoints.append(point_set)
            bcat.append(cat)
        bpoints = np.array(bpoints)
        
        self.log.debug('iter - output data shape: {}'.format(bpoints.shape))
        # bpoints = np.concatenate(bpoints, axis=0)
        return bpoints, np.array(bcat)


if __name__ == "__main__":
    root = '/home/lebesgue/data/Kitti/modelnet40_normal_resampled'
    resource_file = 'modelnet10_train.txt'
    category_file = 'modelnet10_shape_names.txt'
    LOG_FORMAT = "%(asctime)s-%(levelname)s-[%(name)s]:%(message)s"
    lg.basicConfig(filename='data_test.log', level=lg.DEBUG, format=LOG_FORMAT)    

    loader = modelnetload(root, resource_file, category_file, data_augmentation=True)
    data, cls = loader.iter(20, takesample=True)
    print(data.shape)
    print(cls)


    