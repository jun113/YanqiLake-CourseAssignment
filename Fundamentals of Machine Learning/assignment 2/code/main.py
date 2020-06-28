import mxnet as mx
from sklearn import manifold
from assignment_dataset import assigment_dataset
from ML_model import K_Means,nn_3_layer
from matplotlib import pyplot as plt 
#pd.set_option('display.max_rows', None)
#np.set_printoptions(threshold=1e7)
#----------------------------------
#TODO debug current function

#------------------------------------
if __name__ == "__main__":
    is_savefig=True
    dataset = assigment_dataset(savefig=is_savefig) 

    dataset.v2_1()
    model1=K_Means(x_data=dataset.x_data,y_data=dataset.y_data,labels_num=dataset.labels_num,ctx=mx.gpu())
    center_pred=model1.v1(savefig=is_savefig,save_path=dataset.result_path)
    model1.reuslt()

    model2=nn_3_layer(x_data=dataset.x_data,y_data=dataset.y_data,labels_num=dataset.labels_num,train_percent=60,ctx=mx.gpu())
    net=model2.net(l1_num=4,l2_num=3)
#    model2.training(net=net,batch_size=32,lr=0.35,num_epochs=50)
    model2.training_arg(save_path=dataset.result_path)

    dataset.v2_2()
    plt.title('Laplacian Eigenmapping')
    for i in range(1,5):
        trans_data_LE=manifold.SpectralEmbedding(n_components=2,n_neighbors=i*5).fit_transform(dataset.x_data.T)

        plt.subplot(2,2,i)
        plt.scatter(trans_data_LE[:, 0], trans_data_LE[:, 1], c=dataset.corlor_list,marker='o',cmap=plt.cm.hot)
    plt.savefig(dataset.result_path+'\\le_result.png')
    plt.close()

    plt.title('Locally linear embedding')
    for i in range(1,5):
        trans_data_LLE = manifold.LocallyLinearEmbedding(n_components = 2,n_neighbors =i*5,method='standard').fit_transform(dataset.x_data.T)

        plt.subplot(2,2,i)
        plt.scatter(trans_data_LLE[:, 0], trans_data_LLE[:, 1], c=dataset.corlor_list,marker='o',cmap=plt.cm.hot)
    plt.savefig(dataset.result_path+'\\lle_result.png')
    plt.close()

    plt.title('Isometric feature mapping')
    for i in range(1,5):
        trans_data_ISOMAP=manifold.Isomap(n_components=2,n_neighbors=i*5).fit_transform(dataset.x_data.T)

        plt.subplot(2,2,i)
        plt.scatter(trans_data_ISOMAP[:, 0], trans_data_ISOMAP[:, 1], c=dataset.corlor_list,marker='o',cmap=plt.cm.hot)
    plt.savefig(dataset.result_path+'\\isomap_result.png')
    plt.close()
