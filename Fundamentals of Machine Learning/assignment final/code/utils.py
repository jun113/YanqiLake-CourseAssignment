import gzip, time, random, os, os.path as path
import numpy as np, pandas as pd
from matplotlib import pyplot as plt

def load_dataset(dataset_name):
    '''
    return numpy: x_train, y_train, x_test, y_test
    MNIST: http://yann.lecun.com/exdb/mnist/
    Letter Recognition: http://archive.ics.uci.edu/ml/datasets/Letter+Recognition
    '''
    PATH = path.abspath(path.join(path.dirname("__file__"),'dataset'))

    if dataset_name.title() == 'Letter Recognition':
        print('loading dataset: Letter Recognition')
        data_path = path.join(PATH, dataset_name.title(), 'letter-recognition.data')

        x_data = np.loadtxt(fname=data_path, dtype=float, delimiter=',', usecols=range(1,17))
        y_data = np.loadtxt(fname=data_path, dtype=str, delimiter=',', usecols=0)

        y_data = np.array([float(ord(y_data[i])-ord('A')) for i in range(len(y_data))])

        x_train = x_data[:-4000,:]
        y_train = y_data[:-4000]
        x_test = x_data[-16000:,:]
        y_test = y_data[-16000:]

        return x_train, y_train, x_test, y_test

    elif dataset_name.upper() == 'MNIST':
        print('loading dataset: MNIST')

        TRAIN_IMAGES = path.join(PATH, dataset_name.upper(), 'train-images-idx3-ubyte.gz')
        TRAIN_LABELS = path.join(PATH, dataset_name.upper(), 'train-labels-idx1-ubyte.gz')
        TEST_IMAGES = path.join(PATH, dataset_name.upper(), 't10k-images-idx3-ubyte.gz')
        TEST_LABELS = path.join(PATH, dataset_name.upper(), 't10k-labels-idx1-ubyte.gz')   

        x_train = load_MNIST_img(TRAIN_IMAGES)
        y_train = load_MNIST_label(TRAIN_LABELS)
        x_test = load_MNIST_img(TEST_IMAGES)
        y_test = load_MNIST_label(TEST_LABELS)

        return x_train, y_train, x_test, y_test
    
def load_MNIST_img(file_path):
    with gzip.open(file_path, 'rb') as bytestream:
        img_data=np.frombuffer(bytestream.read(), np.uint8, offset=16).astype(np.float32)
    #normalize : 将图像的像素归一化为 0.0~1.0
        img_data /= 255.0
    return img_data.reshape(-1, 784)

def load_MNIST_label(file_path):
    with gzip.open(file_path, 'rb') as bytestream:
        label_data=np.frombuffer(bytestream.read(), np.uint8, offset=8)
    return label_data

def randomcolor():
    '''
    生成随机颜色
    '''
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ''
    for _ in range(6):
        color += colorArr[random.randint(0,14)]
    return "#" + color

def mkdir():
    '''
    创建以当前时间为文件名的文件夹，并返回该文件夹地址
    '''
    current_time = time.strftime("%Y-%m-%d %H%M%S", time.localtime())
    PATH = path.abspath(path.join(path.dirname("__file__"), path.pardir)) + '\\output'
    if not path.exists(PATH):
        print('creating folder...')
        os.mkdir(PATH)
        print(PATH)
        PATH += '\\' + current_time
        os.mkdir(PATH)
        print(PATH)
        print('Done.')
    else:
        PATH += '\\' + current_time
        if not path.exists(PATH):
            print('creating folder...')
            os.mkdir(PATH)
            print(PATH)
            print('Done.')
        else:
            print('the folder has been created.')
            print(PATH)
    return PATH

def output_assignment_1(data, dataset_name, n_components, n_neighbors, save_path):

    data_np = np.array(data)
    result_pd = pd.DataFrame(data_np, index=n_components, columns=n_neighbors)
    
    print(result_pd,'\n')

    title = dataset_name + ' Accuracy Curve'

    plt.title(title)
    plt.plot(n_components, data_np[:,0], c=randomcolor(), marker='.')
    plt.plot(n_components, data_np[:,1], c=randomcolor(), marker='*')
    plt.xlabel('PCA_dim')
    plt.ylabel('Acc')

    plt.xticks(n_components)
    plt.legend(['1-NN Acc','3-NN Acc'])
    plt.grid()

    for x, y in zip(n_components, data_np[:,0]):
        plt.text(x, y, '%.3f'%(y), fontdict={'fontsize': 9})
    for x, y in zip(n_components, data_np[:,1]):
        plt.text(x, y, '%.3f'%(y), fontdict={'fontsize': 9})

    if save_path != None:
        save_name_fig = title + '.png'
        save_name_excel = dataset_name + '_PCA_LDA_KNN.xlsx'             
        result_pd.to_excel(path.join(save_path, save_name_excel))
        plt.savefig(path.join(save_path, save_name_fig))
    else:
        plt.show()
    plt.close()

def output_assignment_2(acc_hidden, loss_hidden, acc_lr, loss_lr, acc_batch_size, loss_batch_size, \
        default_num_hidden, default_lr, default_batch_size, \
        dataset_name, feature_dim, num_labels_list, num_hidden_list, lr_list, batch_size_list, save_path):

    num_epochs = len(acc_hidden[0])

    string = str(num_hidden_list)
    #hidden dim acc
    title = dataset_name + ' Test Accuracy Curve(hidden dim)'

    plt.title(title)
    for i in range(len(num_hidden_list)):
        plt.plot(range(1, num_epochs+1), acc_hidden[i], c=randomcolor(), label='hidden dim: %d'%(num_hidden_list[i]*feature_dim))
    plt.xlabel('epoch')
    plt.ylabel('Acc')
    plt.grid()
    plt.legend()
    if save_path != None:
        save_name_fig = title + '.png'
        save_name_txt = title + '.txt'
        plt.savefig(path.join(save_path, save_name_fig))

        head = title + '\nnum_hidden: '+ string[1:-1] + '\n' + \
            'num_epochs=%d, batch_size=%d, lr=%.3f'%(num_epochs, default_batch_size, default_lr)
        np.savetxt(path.join(save_path, save_name_txt), np.array(acc_hidden), fmt='%.5f', delimiter=',', header=head)
    else:
        plt.show()
    plt.close()

    #hidden dim loss
    title = dataset_name + ' Train Loss Curve(hidden dim)'
    plt.title(title)
    for i in range(len(num_hidden_list)):
        plt.plot(range(1, num_epochs+1), loss_hidden[i], c=randomcolor(), label='hidden dim: %d'%(num_hidden_list[i]*feature_dim))
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    if save_path != None:
        save_name_fig = title + '.png'
        save_name_txt = title + '.txt'
        plt.savefig(path.join(save_path, save_name_fig))

        head = title + '\nnum_hidden: '+ string[1:-1] + '\n' + \
            'num_epochs=%d, batch_size=%d, lr=%.3f'%(num_epochs, default_batch_size, default_lr)
        np.savetxt(path.join(save_path, save_name_txt), np.array(acc_hidden), fmt='%.5f', delimiter=',', header=head)
    else:
        plt.show()
    plt.close()

    string = str(lr_list)
    #lr acc
    title = dataset_name + ' Test Accuracy Curve(lr)'
    plt.title(title)
    for i in range(len(lr_list)):
        plt.plot(range(1, num_epochs+1), acc_lr[i], c=randomcolor(), label='lr: %.3f'%(lr_list[i]))
    plt.xlabel('epoch')
    plt.ylabel('Acc')
    plt.grid()
    plt.legend()
    if save_path != None:
        save_name_fig = title + '.png'
        save_name_txt = title + '.txt'
        plt.savefig(path.join(save_path, save_name_fig))

        head = title + '\nlr: '+ string[1:-1] + '\n' + \
            'num_epochs=%d, batch_size=%d, hidden_dim=%d'%(num_epochs, default_batch_size, int(default_num_hidden*feature_dim))
        np.savetxt(path.join(save_path, save_name_txt), np.array(acc_hidden), fmt='%.5f', delimiter=',', header=head)
    else:
        plt.show()
    plt.close()

    #lr loss
    title = dataset_name + ' Train Loss Curve(lr)'
    plt.title(title)
    for i in range(len(lr_list)):
        plt.plot(range(1, num_epochs+1), loss_lr[i], c=randomcolor(), label='lr: %.3f'%(lr_list[i]))
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    if save_path != None:
        save_name_fig = title + '.png'
        save_name_txt = title + '.txt'
        plt.savefig(path.join(save_path, save_name_fig))

        head = title + '\nlr: '+ string[1:-1] + '\n' + \
            'num_epochs=%d, batch_size=%d, hidden_dim=%d'%(num_epochs, default_batch_size, int(default_num_hidden*feature_dim))

        np.savetxt(path.join(save_path, save_name_txt), np.array(acc_hidden), fmt='%.5f', delimiter=',', header=head)
    else:
        plt.show()
    plt.close()

    string = str(num_hidden_list)
    #batch_size acc
    title = dataset_name + ' Test Accuracy Curve(batch_size)'
    plt.title(title)
    for i in range(len(batch_size_list)):
        plt.plot(range(1, num_epochs+1), acc_batch_size[i], c=randomcolor(), label='batch_size: %d'%(batch_size_list[i]))
    plt.xlabel('epoch')
    plt.ylabel('Acc')
    plt.grid()
    plt.legend()
    if save_path != None:
        save_name_fig = title + '.png'
        save_name_txt = title + '.txt'
        plt.savefig(path.join(save_path, save_name_fig))

        head = title + '\nlr: '+ string[1:-1] + '\n' + \
            'num_epochs=%d, lr=%.3f, hidden_dim=%d'%(num_epochs,  default_lr, int(default_num_hidden*feature_dim))
        np.savetxt(path.join(save_path, save_name_txt), np.array(acc_hidden), fmt='%.5f', delimiter=',', header=head)
    else:
        plt.show()
    plt.close()


    string = str(batch_size_list)
    #batch_size loss
    title = dataset_name + ' Train Loss Curve(batch_size)'
    plt.title(title)
    for i in range(len(batch_size_list)):
        plt.plot(range(1, num_epochs+1), loss_batch_size[i], c=randomcolor(), label='batch_size: %d'%(batch_size_list[i]))
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    if save_path != None:
        save_name_fig = title + '.png'
        save_name_txt = title + '.txt'
        plt.savefig(path.join(save_path, save_name_fig))

        head = title + '\nlr: '+ string[1:-1] + '\n' + \
            'num_epochs=%d, lr=%.3f, hidden_dim=%d'%(num_epochs,  default_lr, int(default_num_hidden*feature_dim))
        np.savetxt(path.join(save_path, save_name_txt), np.array(acc_hidden), fmt='%.5f', delimiter=',', header=head)
    else:
        plt.show()
    plt.close()