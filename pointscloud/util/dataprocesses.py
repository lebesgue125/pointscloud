import numpy as np

def generate_onehot(cls_vec, class_num):
    size = len(cls_vec)
    one_hot = np.zeros(shape=(size, class_num))
    one_hot[range(size), cls_vec] = 1
    return one_hot


if __name__ == "__main__":
    cls_vec = np.array([1,2,3,4,5,6,7,8,9,10])
    class_num = 20
    print(class_onehot(cls_vec, class_num))
