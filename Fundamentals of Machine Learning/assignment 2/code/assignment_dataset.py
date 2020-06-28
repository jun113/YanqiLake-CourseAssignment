import matplotlib.pyplot as plt
import random
import numpy as np
import os
import os.path as path
import time
import pandas as pd
from process_data import _init_dataset,_preprocessing,_final_dataset
from colors import randomcolor
from mpl_toolkits.mplot3d import proj3d
class assigment_dataset():
    def __init__(self,savefig):
        if savefig : self.result_path=self.mkdir()
        else :self.result_path=None

    def v1(self):
        arg={}
        arg['mu']=[]
        arg['mu'].append(np.array([1,-1]))
        arg['mu'].append(np.array([5,-4]))
        arg['mu'].append(np.array([1,4]))
        arg['mu'].append(np.array([6,4.5]))
        arg['mu'].append(np.array([7.5,0.0]))
        arg['Sigma']=np.array([[1,0],[0,1]])
        arg['labels_num']=5
        arg['num']=200
        self.x_data,self.y_data=self.init_data(_init_dataset,arg)
    def v2_1(self):
        arg={}
        arg['mu']=[]
        arg['mu'].append(np.array([1,-1]))
        arg['mu'].append(np.array([5,-4]))
        arg['mu'].append(np.array([1,4]))
        arg['mu'].append(np.array([6,4]))
        arg['mu'].append(np.array([7,0.0]))
        arg['Sigma']=np.array([[1,0],[0,1]])
        arg['labels_num']=5
        arg['num']=200
        self.x_data,self.y_data=self.init_data(_init_dataset,arg)
    def v2_2(self):
        N=1000
        noise=0.001*np.random.randn(N)
        tt=(3*np.pi/2)*(1+2*np.random.rand(N))
        height=21*np.random.rand(N)
        X=np.array([(tt+noise)*np.cos(tt),height,(tt+noise)*np.sin(tt)])
        self.x_data=X
        self.y_data=None
        self.corlor_list=np.squeeze(tt)
        if self.result_path is not None : 
            #TODO
            ax1=plt.axes(projection='3d')
            ax1.scatter3D(X[0,:], X[1,:], X[2,:],c=np.squeeze(tt), cmap=plt.cm.hot)
            plt.title('Swiss Roll')
            plt.savefig(self.result_path+'\\dataset-2.2.png')
            plt.close()


    def init_data(self,func,arg):
        print('init dataset...')
        init_dataset,self.labels_num=func(arg)
        if self.result_path is not None: self.draw(init_dataset)
        else: print('done.')
        return _preprocessing(init_dataset)
    def mkdir(self):
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

    def draw(self,dataset):
        label=[]
        for i in range(self.labels_num):
            plt.scatter(dataset[i]['x'],dataset[i]['y'],c=randomcolor(),marker='.')
            label.append('x'+format(i))

        plt.title('init dataset')
        plt.legend(label)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid()
        plt.savefig(self.result_path+'\\dataset.png')
        plt.close()

    def run(self,model):
        model(self.x_data,self.labels_num)
        pass
        
    def debug(self):
        print('-------start debug---------------')

        print('-------end--------------')