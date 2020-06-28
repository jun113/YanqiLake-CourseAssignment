import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx
import time
from mxnet import nd,autograd,gluon,init
from mxnet.gluon import  loss as gloss, nn, utils as gutils
from process_dataset import load_dataset_to_mxnet,mkdir,randomcolor
class model():
    def __init__(self,x_train,y_train,x_test,y_test,labels_num,ctx):
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        self.labels_num=labels_num
        self.ctx=ctx
        self.PATH=mkdir()

class Net(model):
    def __init__(self, x_train, y_train, x_test, y_test, labels_num, ctx):
        super().__init__(x_train, y_train, x_test, y_test, labels_num, ctx)
        self.loss_func=gloss.SoftmaxCrossEntropyLoss()
        self.net=None

    def LeNet(self):
        '''
        实现经典的LeNet
        第一层，卷积层，输出通道：6，卷积核：5*5，激活函数：sigmoid；
        第二层，池化层，窗口形状：2*2，步幅：2；
        第三层，卷积层，输出通道：16，卷积核：5*5，激活函数：sigmoid；
        第四层，池化层，窗口形状：2*2，步幅：2；
        第五层，全连接层，输出个数：120，激活函数：sigmoid；
        第六层，全连接层，输出个数：84，激活函数：sigmoid；
        第七层，输出层，输出个数：10(标签数)；
        '''
        net=nn.Sequential()
        net.add(nn.Conv2D(channels=6,kernel_size=5,activation='sigmoid'),\
                nn.MaxPool2D(pool_size=2,strides=2),\
                nn.Conv2D(channels=16,kernel_size=5,activation='sigmoid'),\
                nn.MaxPool2D(pool_size=2,strides=2),\
                nn.Dense(120,activation='sigmoid'),\
                nn.Dense(84,activation='sigmoid'),\
                nn.Dense(self.labels_num))
        print('init net...')
        print('--------------------')
        print(net)
        print('--------------------')
        net.collect_params().initialize(force_reinit=True, ctx=self.ctx)
        net.initialize(force_reinit=True,ctx=self.ctx,init=init.Xavier())
        return net

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

    def training(self, net, batch_size, lr,num_epochs,istraining=True):
        """Train and evaluate a model with CPU or GPU."""
        if istraining:  print('training on', self.ctx)
        features_train=nd.array(self.x_train,ctx=self.ctx)
        labels_train=nd.array(self.y_train,ctx=self.ctx)
        features_test=nd.array(self.x_test,ctx=self.ctx)
        labels_test=nd.array(self.y_test,ctx=self.ctx)

        train_iter=load_dataset_to_mxnet(features_train,labels_train,batch_size)
        test_iter=load_dataset_to_mxnet(features_test,labels_test,batch_size)
        trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})

        test_acc= 0.0
        test_acc_list=[]
        train_acc=[]
        train_acc_list=[]
        start_time=time.time()
        for epoch in range(num_epochs):
            train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
            #训练
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
                train_acc_sum += (temp1 == temp2).sum().asscalar()
                n += y.size
                train_acc.append(train_acc_sum/n)
            #测试
            test_acc = self.evaluate_accuracy(test_iter, net, ctx=self.ctx)
            train_acc_list.append(np.mean(train_acc))
            test_acc_list.append(test_acc)
            if istraining:
                print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
                     'time %.1f sec'
                     % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                    time.time() - start))
        if istraining is False:
            print('train acc: %.3f, test acc: %.3f, time: %.2f sec'
            %(np.mean(train_acc),test_acc,time.time() - start_time))
        #保存训练过程 （准确率变化）
        if istraining:  
            plt.title('Accuracy Curve')
            plt.plot(range(1,num_epochs+1),train_acc_list,c=randomcolor(),label='train acc')
            plt.plot(range(1,num_epochs+1),test_acc_list,c=randomcolor(),label='test acc')
#            plt.xticks(range(num_epochs+1))
#            plt.yticks(range(0,1,0.05))
            plt.ylabel('Acc')
            plt.xlabel('epoch')
            plt.grid()
            plt.legend()
            plt.savefig(self.PATH+'\\acc-curve.png')
            plt.close()
            self.net=net

            return net
        else:
            return test_acc

    def training_arg(self,net):
        '''
        训练超参数
        '''
        print('****************')
        print('start training hyperparameters.')
        print('training on', self.ctx)
        num_epochs=10
        lr_acc=[]
        lr_list=[i*0.15 for i in range(1,11)]

        for i in range(10):
            print('learning rate: %.3f'%(lr_list[i]))
            lr_acc.append(self.training(net=net,batch_size=64,lr=lr_list[i],num_epochs=num_epochs,istraining=False))

        print('train lr done.')
        best_lr= lr_list[lr_acc.index(max(lr_acc))]
        print('the best learning rate: {}'.format(best_lr))

        plt.title('Acc Curve')
        plt.plot(lr_list,lr_acc,c=randomcolor())
        plt.xticks(lr_list)
        plt.yticks(lr_acc)
        plt.ylabel('Acc')
        plt.grid()
        plt.xlabel('learning rate')
        plt.savefig(self.PATH+'\\lr-acc.png')
        plt.close()

        batch_size_acc=[]
        batch_size_list=[(2**i)*32 for i in range(5)]

        for i in range(5):
            print('batch size: %.3f'%(batch_size_list[i]))
            batch_size_acc.append(self.training(net=net,\
                        batch_size=batch_size_list[i],\
                        lr=best_lr,\
                        num_epochs=num_epochs,istraining=False))

        print('train size of batch done.')
        best_batch_size= batch_size_list[batch_size_acc.index(max(batch_size_acc))]
        print('the best batch_size: {}'.format(best_batch_size))

        plt.title('Acc Curve')
        plt.plot(batch_size_list,batch_size_acc,c=randomcolor())
        plt.xticks(batch_size_list)
        plt.yticks(batch_size_acc)
        plt.ylabel('Acc')
        plt.grid()
        plt.xlabel('size of batch')
        plt.savefig(self.PATH+'\\batch_size-acc.png')
        plt.close()
        print('Done')        

        return best_lr,best_batch_size

    def test_pred_MNIST(self):
        '''
        测试训练成果，从测试集中随机抽取9张图片
        '''
        if self.net is None:
            print('The net has not been trained!')
        else:
            pred_labels=np.ones((3,3))
            true_labels=np.ones((3,3))
            index=np.random.randint(0,len(self.x_test[:,0,0,0]),9)
            nd_images=nd.array(self.x_test[index,0,:,:],ctx=self.ctx)
            nd_images=nd.reshape(nd_images,(-1,1,28,28))
            result=self.net(nd_images).argmax(axis=1)
            for i in range(3):
                for j in range(3):
                    pred_labels[i,j]=result[i*3+j].asscalar().astype(np.uint8)
                    true_labels[i,j]=self.y_test[index[i*3+j]]
                    plt.subplot(3,3,i*3+j+1)
                    plt.imshow(self.x_test[index[i*3+j],0,:,:], \
                        cmap=plt.get_cmap('gray_r'))
                    plt.axis('off')
            plt.savefig(self.PATH+'\\test_pred_MNIST.png')
            plt.close()
            print('test result.')
            print('true labels:')
            print(true_labels)
            print('pred labels:')
            print(pred_labels)
