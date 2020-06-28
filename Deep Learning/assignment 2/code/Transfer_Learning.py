from mxnet.gluon import  loss as gloss, nn, utils as gutils,model_zoo
import time
from mxnet import nd,autograd,gluon,init
import mxnet as mx
import numpy as np
from process_dataset import load_img_batch
from auxiliary_func import randomcolor
import d2lzh as d2l
from matplotlib import pyplot as plt


def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()],ispred=True):
    """Evaluate accuracy of a model on the given data set."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    if ispred: 
        pred_list=[]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            n += y.size
            y = y.astype('float32')

            output_features=net.features(X.as_in_context(ctx[0]))
            output = nd.softmax(net.output_new(output_features))

            pred=nd.reshape(output.argmax(axis=1),(-1))
            true=nd.reshape(y,(-1))
            acc_sum += (pred == true).sum().copyto(mx.cpu())

            if ispred:
                pred_list+=pred.asnumpy().astype(np.int).tolist()

        acc_sum.wait_to_read()

    if ispred:
        print('test set acc: %.3f'%(acc_sum.asscalar()/n))
        return np.reshape(pred_list,(-1))
    else:
        return acc_sum.asscalar() / n

def _get_batch(batch, ctx):
    """Return features and labels on ctx."""
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])

def get_net(labels_num,ctx):
    finetune_net=model_zoo.vision.resnet34_v2(pretrained=True)

    #定义新的输出层
    finetune_net.output_new=nn.HybridSequential(prefix='')
    finetune_net.output_new.add(\
        nn.Dense(256,activation='relu'),\
        nn.Dense(labels_num))
    #初始化输出层参数
    finetune_net.output_new.initialize(init.Xavier(),ctx)
    #将网络参数分配到相应设备上
    finetune_net.collect_params().reset_ctx(ctx)
    return finetune_net


def train(net, train_data, validation_data, batch_size, lr, wd, num_epochs, ctx,PATH):

    train_iter=load_img_batch(train_data,batch_size,'train')
    validation_iter=load_img_batch(validation_data,batch_size,'train')
    
    loss_func=gloss.SoftmaxCrossEntropyLoss()
    trainer=gluon.Trainer(net.output_new.collect_params(),'sgd',{'learning_rate': lr, 'momentum': 0.9,'wd': wd})

    test_acc= 0.0
    test_acc_list=[]
    train_acc=[]
    train_acc_list=[]
    train_loss_list=[]
    start_time=time.time()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        #训练
        for X, y in train_iter:

            y=y.as_in_context(ctx) 
            output_features=net.features(X.as_in_context(ctx))
            with autograd.record():
                outputs=net.output_new(output_features)
                l=loss_func(outputs,y).sum()
            l.backward()
            trainer.step(batch_size)
            
            train_l_sum+=l.asscalar()
            n+=y.size
        #测试
        test_acc=evaluate_accuracy(validation_iter, net, ctx=ctx, ispred=False)

        train_loss_list.append(train_l_sum/n)
        test_acc_list.append(test_acc)

        print('epoch %d, train loss: %.3f, test acc: %.3f, time: %.2f sec'
        %(epoch+1, train_l_sum/n, test_acc,time.time() - start_time))

    #绘制 训练损失函数 验证集测试准确度
    plt.title('Loss Curve')
    plt.plot(range(1,num_epochs+1),train_loss_list,c=randomcolor(),label='train loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend()
    plt.savefig(PATH+'\\loss-curve.png')
    plt.close()

    plt.title('Accuracy Curve')
    plt.plot(range(1,num_epochs+1),test_acc_list,c=randomcolor(),label='validation set acc')
    plt.ylabel('Acc')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend()
    plt.savefig(PATH+'\\acc-curve.png')
    plt.close()

    return net

def predict(net,test_imgs,ctx,save_fig_path):

    test_iter=load_img_batch(test_imgs,64,'test')

    num_imgs=len(test_imgs)
    #从测试集中猫、狗各取三个样本，用于实测
    index_cat=np.random.randint(0,num_imgs/2,size=3)
    index_dog=np.random.randint(num_imgs/2,num_imgs,size=3)
    index=np.append(index_cat,index_dog)
    np.random.shuffle(index)

    pred_list=evaluate_accuracy(test_iter,net,ctx,ispred=True)
    pred_label=[pred_list[i] for i in index]
    true_label=[test_imgs[i][1] for i in index]
    print('cat: 0, dog: 1')
    print('true label:')
    print(np.reshape(true_label,(2,-1)))
    print('predict label:')
    print(np.reshape(pred_label,(2,-1)))

    show_img=[test_imgs[i][0] for i in index]
    d2l.show_images(show_img,2,3,scale=1)
    plt.savefig(save_fig_path+'\\cat_dog.png')
    plt.show()
    plt.close()
