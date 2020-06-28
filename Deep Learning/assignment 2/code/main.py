from auxiliary_func import try_gpu,mkdir
from process_dataset import load_dataset
from Transfer_Learning import get_net,train,predict

if __name__ == '__main__':
    save_fig_path=mkdir()
    #类别数
    num_labels=2
    #尝试使用GPU
    ctx=try_gpu()
    #加载数据集
    net=get_net(num_labels,ctx)
    train_ds=load_dataset('kaggle','train')
    validation_ds=load_dataset('kaggle','validation')

    trained_net=train(net,train_ds,validation_ds,128,0.3,1e-4,25,ctx,save_fig_path)
    test_ds=load_dataset('kaggle','test')
    predict(trained_net,test_ds,ctx,save_fig_path)    