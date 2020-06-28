import d2lzh as d2l
from DL_model import Net
from process_dataset import load_mnist

if __name__ == '__main__':
    #尝试使用GPU
    ctx=d2l.try_gpu()
    #加载数据集
    train_images,train_labels,test_images,test_labels=load_mnist()
    #初始化模型
    model=Net(x_train=train_images,y_train=train_labels,\
            x_test=test_images,y_test=test_labels,\
            labels_num=10,ctx=ctx)
    #生成网络
    LeNet=model.LeNet()
    #训练超参数
    best_lr,best_batch_size=model.training_arg(net=LeNet)
    #训练模型
    LeNet=model.training(net=LeNet,batch_size=best_batch_size,lr=best_lr,num_epochs=25,istraining=True)
    #测试效果
    model.test_pred_MNIST()