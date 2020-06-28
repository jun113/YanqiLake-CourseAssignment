import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx
import pandas as pd
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import time
from mxnet import nd,autograd,gluon,init
from mxnet.gluon import  loss as gloss, nn, utils as gutils
from process_data import _final_dataset,_load_dataset_mxnet
from colors import randomcolor
class model():
    def __init__(self,x_data,y_data,labels_num,ctx):
        self.x_data=x_data
        self.y_data=y_data
        self.labels_num=labels_num
        self.data_num=len(self.x_data)
        self.ctx=ctx

class K_Means(model):
    def __init__(self, x_data, y_data, labels_num, ctx):
        super().__init__(x_data, y_data, labels_num, ctx)
        self.center=self.init_center()
        self.y_pred=None
        self.acc=None
        self.nmi=None

    def init_center(self):
        return self.x_data[np.random.randint(0,self.data_num,self.labels_num)]

    def v1(self,savefig,save_path):
        print('starting K-Means clustering version 1.')
        center=nd.array(self.center,ctx=self.ctx)
        center_temp=center.zeros_like()
        data=nd.array(self.x_data,ctx=self.ctx)
        #y_pred=self.y_true.zeros_like()
        data=nd.concat(data,nd.zeros((self.data_num,1),ctx=self.ctx),dim=1)
        df=pd.DataFrame(columns=['x','y','label'])
        print('-----------------')
        print('init center:')
        print(center)
        itera=1
        while(itera<=100):
            center_temp[:]=center
            for i in range(self.data_num):
                #delta_x^2 delta_y^2
                delta_xy=(center-data[i,0:-1])**2
                #delta_x^2+delta_y^2
                d_array=delta_xy[:,0]+delta_xy[:,1]
                #return min index
                data[i,2]=nd.argmin(d_array,axis=0)
                #y_pred=nd.argmin(d_array,axis=0)
            #TODO
            df=pd.DataFrame(data.asnumpy(),columns=['x','y','label'])
            for i in range(self.labels_num):
                cluster_temp=df.loc[df['label']==i].values
                center[i]=nd.array(cluster_temp[:,:-1].mean(axis=0),ctx=self.ctx)
            #----------
            itera+=1
            if (center==center_temp).sum().asscalar()==2*self.labels_num: break
        print('pred center:')
        print(center)
        print('iteration:',format(itera))
        print('-----------------')
        if savefig : self.draw(pred_data=df,save_path=save_path)
        self.y_pred=data[:,2].asnumpy()
        return center.asnumpy()

    def v2(self):
        print('starting K-Means clustering version 2.')
        center=nd.array(self.center,ctx=mx.gpu())
        data=nd.array(self.x_data,ctx=mx.gpu())
        data=nd.concat(data,nd.zeros((self.data_num,1),ctx=mx.gpu()),dim=1)
        print('init center:')
        print(center)

    def reuslt(self):
        self.ACC()
        self.NMI()

    def ACC(self):
        y_true = self.y_data.astype(np.int64)
        y_pred = self.y_pred.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        row_ind, col_ind=linear_sum_assignment(w.max() - w)
        self.acc = sum(w[row_ind, col_ind]) / self.data_num 
        print('ACC:',format(self.acc))

    def NMI(self):
        self.nmi = metrics.normalized_mutual_info_score(self.y_data.reshape((-1)),self.y_pred.reshape((-1)))
        print('NMI:',format(self.nmi))
    #TODO    
    def draw(self,pred_data,save_path):
        label=[]
        true_data=pd.DataFrame(self.x_data,columns=list('xy'))
        true_data['label']=self.y_data
        err_dot=[]
        color=[]
        for i in range(self.labels_num): color.append(randomcolor())
        plt.subplot(1,2,1)
        for i in range(self.labels_num):
            x=true_data.loc[true_data['label']==i].values[:,0]
            y=true_data.loc[true_data['label']==i].values[:,1]
            plt.scatter(x,y,c=color[i],marker='.')
            label.append('x'+format(i))
        plt.title('True clustering')
        plt.legend(label)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid()
        
        plt.subplot(1,2,2)
        for i in range(self.labels_num):
            x=pred_data.loc[pred_data['label']==i].values[:,0]
            y=pred_data.loc[pred_data['label']==i].values[:,1]
            plt.scatter(x,y,c=color[i],marker='.')
            label.append('x'+format(i))

        plt.title('Predict clustering')
        plt.legend(label)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid()

        plt.savefig(save_path+'\\k-means clustering.png')
        plt.close()

class nn_3_layer(model):
    def __init__(self, x_data, y_data, labels_num, train_percent, ctx):
        super().__init__(x_data, y_data, labels_num, ctx)
        self.loss_func=gloss.SoftmaxCrossEntropyLoss()

        self.x_train,self.y_train,self.x_test,self.y_test=_final_dataset(self.x_data,self.y_data,train_percent)

    def net(self,l1_num,l2_num):
        net=nn.Sequential()
        net.add(nn.Dense(l1_num,activation='sigmoid'),nn.Dense(l2_num,activation='sigmoid'),nn.Dense(5))
        net.collect_params().initialize(force_reinit=True, ctx=self.ctx)
        net.initialize(force_reinit=True,ctx=self.ctx,init=init.Xavier())
        return net


    def training(self, net, batch_size, lr,num_epochs):
        """Train and evaluate a model with CPU or GPU."""
#        print('training on', self.ctx)
        features_train=nd.array(self.x_train,ctx=self.ctx)
        labels_train=nd.array(self.y_train,ctx=self.ctx)
        features_test=nd.array(self.x_test,ctx=self.ctx)
        labels_test=nd.array(self.y_test,ctx=self.ctx)

        train_iter=_load_dataset_mxnet(features_train,labels_train,batch_size)
        test_iter=_load_dataset_mxnet(features_test,labels_test,batch_size)
        trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})

        test_acc= 0.0
        train_acc=[]
        start_time=time.time()
        for epoch in range(num_epochs):
            train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
            for X, y in train_iter:
                X, y = X.as_in_context(self.ctx), y.as_in_context(self.ctx)
                with autograd.record():
                    y_hat = net(X)
                    l = self.loss_func(y_hat, y).sum()
                l.backward()
                trainer.step(batch_size)
                y = y.astype('float32')
                train_l_sum += l.asscalar()
                temp1=nd.reshape(y_hat.argmax(axis=1),(-1))
                temp2=nd.reshape(y,(-1))
                #train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
                train_acc_sum += (temp1 == temp2).sum().asscalar()
                n += y.size
                train_acc.append(train_acc_sum/n)

            test_acc = self.evaluate_accuracy(test_iter, net, ctx=self.ctx)
            '''
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
                 'time %.1f sec'
                 % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                    time.time() - start))
            '''
        print('train acc: %.3f, test acc: %.3f, time: %.2f'
        %(np.mean(train_acc),test_acc,time.time() - start_time))
        return test_acc
        

    def evaluate_accuracy(self, data_iter, net, ctx=[mx.cpu()]):
        """Evaluate accuracy of a model on the given data set."""
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        acc_sum, n = nd.array([0]), 0
        for batch in data_iter:
            features, labels, _ = self._get_batch(batch, ctx)
            for X, y in zip(features, labels):
                y = y.astype('float32')

                temp1=nd.reshape(net(X).argmax(axis=1),(-1))
                temp2=nd.reshape(y,(-1))
                acc_sum += (temp1 == temp2).sum().copyto(mx.cpu())
#                acc_sum += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
                n += y.size
            acc_sum.wait_to_read()
        return acc_sum.asscalar() / n

    def _get_batch(self,batch, ctx):
        """Return features and labels on ctx."""
        features, labels = batch
        if labels.dtype != features.dtype:
            labels = labels.astype(features.dtype)
        return (gutils.split_and_load(features, ctx),
                gutils.split_and_load(labels, ctx), features.shape[0])


    def training_arg(self,save_path):
        print('****************')
        print('start training hyperparameters.')
        print('training on', self.ctx)
        num_epochs=75
        lr_acc=[]
        lr_list=[i*0.15 for i in range(1,11)]

        for i in range(1,11):
            print('learning rate: %.3f'%(lr_list[i-1]))
            lr_acc.append(self.training(net=self.net(l1_num=4,l2_num=3),batch_size=32,lr=lr_list[i-1],num_epochs=num_epochs))

        print('train lr done.')
        lr= lr_list[lr_acc.index(max(lr_acc))]
        print('the best learning rate: {}'.format(lr))


        l1_node_acc=[]
        l2_node_acc=[]
        for i in range(1,11):
            print('Number of nodes in the hide layer 1: %d'%(i))
            l1_node_acc.append(self.training(net=self.net(l1_num=i,l2_num=3),batch_size=32,lr=lr,num_epochs=num_epochs))
        
        print('layer1 done.')
        l1_node=l1_node_acc.index(max(l1_node_acc))+1
        print('the best layer 1 num: {}'.format(l1_node))

        for i in range(1,11):
            print('Number of nodes in the hide layer 2: %d'%(i))
            l2_node_acc.append(self.training(net=self.net(l1_num=l1_node,l2_num=i),batch_size=32,lr=lr,num_epochs=num_epochs))

        print('layer2 done.')
        l2_node=l2_node_acc.index(max(l2_node_acc))+1
        print('the best layer 2 num: {}'.format(l2_node))
        print('save result...')

        plt.title('Acc Curve')
        plt.plot(lr_list,lr_acc,c=randomcolor())
        plt.xticks(lr_list)
        plt.yticks(lr_acc)
        plt.ylabel('Acc')  # y轴变量名称
        plt.grid()
        plt.xlabel('learning rate')
        plt.savefig(save_path+'\\mxnet_lr-acc.png')
        plt.close()

        plt.title('Acc Curve')
        plt.plot(range(1,11),l1_node_acc,c=randomcolor())
        plt.plot(range(1,11),l2_node_acc,c=randomcolor())
        plt.xticks(range(1,11))
        plt.ylabel('Acc')  # y轴变量名称
        plt.xlabel('num of node')  # y轴变量名称
        plt.grid()

        plt.legend(['layer 1(num of node)','layer 2(num of node)'])
        plt.savefig(save_path+'\\mxnet_node-acc.png')
        plt.close()
        print('Done')        
