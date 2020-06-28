from mxnet.gluon import data as gdata
import os
import os.path as path
import time
import gzip
import numpy as np
import random


def load_mnist():
    '''
    加载本地MNIST数据集
    '''
    PATH=path.abspath(path.join(path.dirname("__file__"),'MNIST'))
    TRAIN_IMAGES=os.path.join(PATH,'train-images-idx3-ubyte.gz')
    TRAIN_LABELS=os.path.join(PATH,'train-labels-idx1-ubyte.gz')
    TEST_IMAGES=os.path.join(PATH,'t10k-images-idx3-ubyte.gz')
    TEST_LABELS=os.path.join(PATH,'t10k-labels-idx1-ubyte.gz')   

    x_train=_load_img(TRAIN_IMAGES)
    y_train=_load_label(TRAIN_LABELS)
    x_test=_load_img(TEST_IMAGES)
    y_test=_load_label(TEST_LABELS)

    return x_train,y_train,x_test,y_test

def _load_img(file_path):
    '''
    加载图片
    '''
    with gzip.open(file_path,'rb') as bytestream:
        img_data=np.frombuffer(bytestream.read(),np.uint8,offset=16).astype(np.float32)
    #normalize : 将图像的像素值正规化为0.0~1.0
    img_data/=255.0
    #index * channels * columns * rows
    img_data=img_data.reshape(-1,1,28,28)
    return img_data

def _load_label(file_path):
    '''
    加载标签
    '''
    with gzip.open(file_path,'rb') as bytestream:
        label_data=np.frombuffer(bytestream.read(),np.uint8,offset=8)
    return label_data
#    return _change_one_hot_label(label_data)

def _change_one_hot_label(X):
    '''
    将标签转为独热码的形式 （这次没用到）
    '''
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T

def load_dataset_to_mxnet(features,labels,batch_size):
    '''
    将特征与标签配对，形成小批量的形式用于训练
    '''
    dataset=gdata.ArrayDataset(features,labels)
    return gdata.DataLoader(dataset,batch_size,shuffle=True)

def mkdir():
    '''
    创建以当前时间为文件名的文件夹，并返回该文件夹地址
    '''
    current_time=time.strftime("%Y-%m-%d %H%M%S", time.localtime())
    PATH=path.abspath(path.join(path.dirname("__file__"),path.pardir))+'\\picture'
    if not path.exists(PATH):
        print('creating folder...')
        os.mkdir(PATH)
        print(PATH)
        PATH+='\\'+current_time
        os.mkdir(PATH)
        print(PATH)
        print('Done.')
    else:
        PATH+='\\'+current_time
        if not path.exists(PATH):
            print('creating folder...')
            os.mkdir(PATH)
            print(PATH)
            print('Done.')
        else:
            print('the folder has been created.')
            print(PATH)
    return PATH

def randomcolor():
    '''
    生成随机颜色
    '''
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ''
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color