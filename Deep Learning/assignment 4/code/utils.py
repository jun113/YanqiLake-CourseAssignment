import time, random, os, os.path as path
import numpy as np, pandas as pd
from matplotlib import pyplot as plt

def load_dataset(data_path, word2id, max_len):
    '''
    return numpy: X, y
    '''
    X, y = [], []
    with open(data_path, encoding='utf-8') as f:
        for line in f.readlines():
            data = line.strip()
            if data == '': continue
            data = data.split()

            st = [word2id.get(word, 0) for word in data[1:]]

            st = st[:max_len] if len(st)>max_len else (max_len-len(st))*[0]+st

            X.append(st)
            y.append(int(data[0]))
    return np.array(X), np.array(y)

def build_word2id(dataset_path, file_word2id):
    if path.exists(file_word2id):
        word2id = {}
        with open(file_word2id, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                word2id[sp[0]] = int(sp[1])
        return word2id
    else:
        word2id = {'_PAD_': 0}
        for dataset in dataset_path:
            with open(dataset, encoding='utf-8') as f:
                for line in f.readlines():
                    sp = line.strip().split()
                    for word in sp[1:]:
                        if word not in word2id.keys():
                            word2id[word] = len(word2id)

        with open(file_word2id, 'w', encoding='utf-8') as f:
            for w in word2id:
                f.write(w+'\t')
                f.write(str(word2id[w]))
                f.write('\n')
        return word2id
def build_word2vec(pretrain_word2vec, word2id, file_word2vec):

    import gensim
    n_words = len(word2id) 
    model = gensim.models.KeyedVectors.load_word2vec_format(pretrain_word2vec, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))

    if path.exists(file_word2vec):
        with open(file_word2vec, encoding='utf-8') as f:
            for i, line in enumerate(f):
                word_vecs[i] = np.array(line.strip().split()).astype(np.float)
    else:

        for word in word2id.keys():
            try:
                word_vecs[word2id[word]] = model[word]
            except KeyError:
                pass
        if file_word2vec:
            with open(file_word2vec, 'w', encoding='utf-8') as f:
                for vec in word_vecs:
                    vec = [str(w) for w in vec]
                    f.write(' '.join(vec))
                    f.write('\n')
    return word_vecs

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
    PATH = path.abspath(path.join(os.getcwd(), path.pardir, 'OUTPUT'))

    if not path.exists(PATH):
        print('creating folder...')
        os.mkdir(PATH)
        print(PATH)
        PATH = path.join(PATH, current_time)
        os.mkdir(PATH)
        print(PATH)
        print('Done.')
    else:
        PATH = path.join(PATH, current_time)
        if not path.exists(PATH):
            print('creating folder...')
            os.mkdir(PATH)
            print(PATH)
            print('Done.')
        else:
            print('the folder has been created.')
            print(PATH)
    return PATH
def output_1(loss_list, acc_list, lr, batch_size, TP, FN, FP, TN, start_time, model_info, save_path):
    num_epochs = len(loss_list)

    plt.title('Training Set: Loss Curve')
    plt.plot(range(1, num_epochs+1), loss_list, c=randomcolor())
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.grid()
    if save_path != None:
        plt.savefig(path.join(save_path, 'Loss Curve.png'))
    else:
        plt.show()
    plt.close()

    plt.title('Test Set: Accuracy Curve')
    plt.plot(range(1, num_epochs+1), acc_list, c=randomcolor())
    plt.xlabel('epoch')
    plt.ylabel('Acc')
    plt.grid()
    if save_path != None:
        plt.savefig(path.join(save_path, 'Accuracy Curve.png'))
    else:
        plt.show()
    plt.close()

    A = (TP+TN)/(TP+FN+FP+TN)
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    S = TN/(TN+FP)
    F1 = 2*(P*R)/(P+R)

    output_string = ['final model\n' + model_info,
    'num_epochs = %d, lr = %.4f, batch_size = %d, total time = %.3f sec' %(num_epochs, lr, batch_size, time.time()-start_time), 
        'Confusion Matrix:', 
        'TP = %d\tFP = %d' %(TP, FP),
        'FN = %d\tTN = %d' %(FN, TN),
        'Accuracy = %.3f\tPrecesion = %.3f\tRecall = %.3f\tSpecificity = %.3f' %(A, P, R, S),
        'F1 Score = %.3f' %(F1)]
    if save_path != None:    
        file_path = path.join(save_path, 'output.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            for content in output_string:
                print(content)
                f.write(content+'\n')
    else:
        for content in output_string:
            print(content)

def output_2(lr_loss, lr_acc, batch_size_loss, batch_size_acc, lr_list, batch_size_list, save_path):

    num_epochs = len(lr_loss[0])

    title = 'Training Set: Loss Curve(lr)'
    plt.title(title)
    for i in range(len(lr_list)):
        plt.plot(range(1, num_epochs+1), lr_loss[i], c=randomcolor(), label='lr: %.4f'%(lr_list[i]))
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    if save_path != None:
        save_name = title.replace(':', ',') + '.png'
        plt.savefig(path.join(save_path, save_name))
    else:
        plt.show()
    plt.close()

    title = 'Validation Set: Accuracy Curve(lr)'
    plt.title(title)
    for i in range(len(lr_list)):
        plt.plot(range(1, num_epochs+1), lr_acc[i], c=randomcolor(), label='lr: %.4f'%(lr_list[i]))
    plt.xlabel('epoch')
    plt.ylabel('Acc')
    plt.grid()
    plt.legend()
    if save_path != None:
        save_name = title.replace(':', ',') + '.png'
        plt.savefig(path.join(save_path, save_name))
    else:
        plt.show()
    plt.close()

    title = 'Training Set: Loss Curve(batch_size)'
    plt.title(title)
    for i in range(len(batch_size_list)):
        plt.plot(range(1, num_epochs+1), batch_size_loss[i], c=randomcolor(), label='batch_size: %.3f'%(batch_size_list[i]))
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    if save_path != None:
        save_name = title.replace(':', ',') + '.png'
        plt.savefig(path.join(save_path, save_name))
    else:
        plt.show()
    plt.close()

    title = 'Validation Set: Accuracy Curve(batch_size)'
    plt.title(title)
    for i in range(len(batch_size_list)):
        plt.plot(range(1, num_epochs+1), batch_size_acc[i], c=randomcolor(), label='batch_size: %.3f'%(batch_size_list[i]))
    plt.xlabel('epoch')
    plt.ylabel('Acc')
    plt.grid()
    plt.legend()
    if save_path != None:
        save_name = title.replace(':', ',') + '.png'
        plt.savefig(path.join(save_path, save_name))
    else:
        plt.show()
    plt.close()