import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mxnet.gluon import data as gdata
from mxnet import nd
import seaborn as sns


#(5,200,3)
def _init_dataset(arg):
    dataset=[]
    for i in range(arg['labels_num']):
        dataset.append(pd.DataFrame(np.random.multivariate_normal(arg['mu'][i],arg['Sigma'],arg['num']),columns=list('xy')))
        dataset[i]['label']=i
    return dataset,arg['labels_num']

# (1000,3)
def _preprocessing(input_data):
    dataset=pd.DataFrame(columns=['x','y','label'])
    for i in range(len(input_data)):
        dataset=dataset.append(input_data[i])
    dataset=np.array(dataset)
#    np.random.seed(10)
#    np.random.shuffle(dataset)
#    x_data,y_data=np.array_split(dataset,2,axis=1)
#    return x_data,y_data
    return np.array_split(dataset,2,axis=1)

def _shuffle_dataset(x_data,y_data):
    index=[i for i in range(len(x_data))]
    np.random.shuffle(index)
    x_data=x_data[index]
    y_data=y_data[index]
    return x_data,y_data 

def _final_dataset(x_data,y_data,train_percent):

    index=[i for i in range(len(x_data))]
    np.random.shuffle(index)
    x_data=x_data[index]
    y_data=y_data[index]

    train_num=int(len(x_data)*train_percent/100)
    x_train=x_data[:train_num]
    y_train=y_data[:train_num]
    x_test=x_data[train_num:]
    y_test=y_data[train_num:]
    return x_train,y_train,x_test,y_test

def _load_dataset_mxnet(features,labels,batch_size):
    dataset=gdata.ArrayDataset(features,labels)
    return gdata.DataLoader(dataset,batch_size,shuffle=False)


        
        