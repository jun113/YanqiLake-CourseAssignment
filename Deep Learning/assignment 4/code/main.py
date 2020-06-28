import time, random, os, os.path as path
import torch
from model import TextCNN, train, model_config, optimal_parameters, scored, pred
from utils import mkdir, build_word2id, build_word2vec, load_dataset, output_1

class config():
    def __init__(self, save_result):
        super().__init__()
        self.save_path = mkdir() if save_result else None
        dataset_path = path.join(os.getcwd(), 'Dataset')
        self.word2id = path.join(dataset_path, 'word2id.txt')
        self.train_set = path.join(dataset_path, 'train.txt')
        self.validation_set = path.join(dataset_path, 'validation.txt')
        self.test_set = path.join(dataset_path, 'test.txt')
        self.pretrain_word2vec = path.join(dataset_path, 'wiki_word2vec_50.bin')
        self.word2vec = path.join(dataset_path, 'word2vec.bin')
        self.max_len = 75
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.lr = [0.0005, 0.001, 0.0015, 0.003, 0.005, 0.01, 0.15]
        self.batch_size = [32, 64, 128, 256, 512]
        self.num_epochs = [15, 30]

def run_1():
    '''
    完整调参过程，费时
    '''
    start_time = time.time()
    #导入配置
    opt = config(True)
    #导入/生成 train/validation dataset word2id
    word2id =  build_word2id([opt.train_set, opt.validation_set], opt.word2id)
    #导入/生成 word2vec
    word2vec = build_word2vec(opt.pretrain_word2vec, word2id, opt.word2vec)
    #导入模型配置
    model_opt = model_config()
    model_opt.pretrained_embed = word2vec
    #导入训练集
    x_train, y_train = load_dataset(opt.train_set, word2id, opt.max_len)
    #导入验证集
    x_validation, y_validation = load_dataset(opt.validation_set, word2id, opt.max_len)
    #调参训练模型
    best_lr, best_batch_size = optimal_parameters(model_opt, x_train, y_train, x_validation, y_validation, opt.lr, opt.batch_size, opt.num_epochs[0], opt.device, opt.save_path)
    print('best lr = %.4f\tbest_batch_size = %d'%(best_lr,best_batch_size))
    #导入测试集    
    x_test, y_test = load_dataset(opt.test_set, word2id, opt.max_len)
    #初始化模型
    model = TextCNN(model_opt)
    #训练模型
    loss_list, acc_list = train(model, x_train, y_train, x_validation, y_validation, best_lr, best_batch_size, opt.num_epochs[1], opt.device)
    #打分
    TP, FN, FP, TN = scored(model, x_test, y_test, best_batch_size, opt.device)
    #准确度达到要求，保存模型
    if (TP+TN)/(TP+FN+FP+TN) >= 0.83:
        os.rename(opt.save_path, opt.save_path + '-mark')
        opt.save_path += '-mark'
        torch.save(model.state_dict(), path.join(opt.save_path, 'model.pytorch'))
    #输出结果
    output_1(loss_list, acc_list, best_lr, best_batch_size, TP, FN, FP, TN, start_time, str(model), opt.save_path)

def run_2():
    '''
    调某个参数
    '''
    #导入配置
    opt = config(True)
    #导入/生成 train/validation dataset word2id
    word2id =  build_word2id([opt.train_set, opt.validation_set], opt.word2id)
    #导入/生成 word2vec
    word2vec = build_word2vec(opt.pretrain_word2vec, word2id, opt.word2vec)
    #导入模型配置
    model_opt = model_config()
    model_opt.pretrained_embed = word2vec
    #导入训练集
    x_train, y_train = load_dataset(opt.train_set, word2id, opt.max_len)
    #导入验证集
    x_validation, y_validation = load_dataset(opt.validation_set, word2id, opt.max_len)
    #导入测试集    
    x_test, y_test = load_dataset(opt.test_set, word2id, opt.max_len)
    #初始化模型
    model = TextCNN(model_opt)
    #训练模型
    loss_list, acc_list = train(model, x_train, y_train, x_validation, y_validation, 0.005, 128, 25, opt.device)
    #打分
    TP, FN, FP, TN = scored(model, x_test, y_test, 128, opt.device)
    #准确度达到要求，保存模型
    if (TP+TN)/(TP+FN+FP+TN) >= 0.83:
        os.rename(opt.save_path, opt.save_path + '-mark')
        opt.save_path += ' %.4f'%((TP+TN)/(TP+FN+FP+TN))
        torch.save(model.state_dict(), path.join(opt.save_path, 'model.pytorch'))
    #输出结果
    print('Acc: %.3f'%((TP+TN)/(TP+FN+FP+TN)))

def run_3(model_path, num):
    #导入训练好的模型，模型实例验证
    opt = config(False)
    word2id =  build_word2id([opt.train_set, opt.validation_set], opt.word2id)
    word2vec = build_word2vec(opt.pretrain_word2vec, word2id, opt.word2vec)
    x_test, y_test = load_dataset(opt.test_set, word2id, opt.max_len)

    model_opt = model_config()
    model = TextCNN(model_opt)

    model.load_state_dict(torch.load(model_path))
    print(model)
    import linecache
    for _ in range(num):
        index = random.randint(0, len(x_test)-1)
        content = linecache.getline(opt.test_set, index).strip()
        
        label_true = content[0]
        label_pred = pred(model, x_test[index])
        content = content[1:]
        print('line: %d'%(index))
        print('content:', content)
        print('true label: '+ label_true + '\tpred label: %d'%(label_pred))

if __name__ == '__main__':
    #run_1()
    run_3('model.pytorch', 5)
