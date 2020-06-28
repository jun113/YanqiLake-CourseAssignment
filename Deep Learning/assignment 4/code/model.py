import time
import numpy as np
from utils import output_2
from torch.nn import init
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch, torch.nn as nn, torch.nn.functional as F

class model_config():

    update_w2v = True           # 是否在训练中更新w2v
    vocab_size = 58954          # 词汇量，与word2id中的词汇量一致
    n_class = 2                 # 分类数：分别为pos和neg
    embedding_dim = 50          # 词向量维度
    drop_keep_prob = 0.3        # dropout层，参数keep的比例
    pretrained_embed = None     # 预训练的词嵌入模型

    kernel_sizes = [3, 4, 5]
    num_kernel = 256
    num_channel = 1

class TextCNN(nn.Module):
    def __init__(self, model_config):
        super(TextCNN, self).__init__()

        update_w2v = model_config.update_w2v
        vocab_size = model_config.vocab_size
        n_class = model_config.n_class
        embedding_dim = model_config.embedding_dim
        drop_keep_prob = model_config.drop_keep_prob
        pretrained_embed = model_config.pretrained_embed

        num_kernel = model_config.num_kernel
        num_channel = model_config.num_channel
        
        # 使用预训练的词向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embed is not None: self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))
        self.embedding.weight.requires_grad = update_w2v
        # 卷积层
        self.convs = nn.ModuleList([nn.Conv2d(num_channel, num_kernel, (K, embedding_dim)) for K in model_config.kernel_sizes])
        # Dropout
        self.dropout = nn.Dropout(drop_keep_prob)
        # 全连接层
        self.fc = nn.Linear(len(model_config.kernel_sizes)*num_kernel, n_class)
        init.xavier_normal(self.fc.weight)
        init.constant(self.fc.bias, 0)

    def forward(self, x):
        x = x.to(torch.int64)
        x = self.embedding(x)
        x = x.unsqueeze(1)
        # len(kernel_size)*(N ,kernel_num, W)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] 
        # len(kernel_size)*(N, kernel_num)
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  

        x = torch.cat(x,1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def train(model, x_train, y_train, x_test, y_test, lr, batch_size, num_epochs, device):

    x_train = torch.from_numpy(x_train).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.long)

    train_set = TensorDataset(x_train, y_train)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

    model.train().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss().to(device)

    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        count, loss_acc, start = 0, 0, time.time()
        # train
        for (feature, label) in train_loader:
            X = Variable(feature).float().to(device)
            y_true = Variable(label).long().to(device)

            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y_true)

            loss.backward()
            optimizer.step()

            count += 1
            loss_acc += loss.item()
        
        # test
        TP, FP, FN, TN= scored(model, x_test, y_test, batch_size, device)
        test_acc = (TP+TN)/(TP+FN+FP+TN)

        # output
        loss_list.append(loss_acc/count)
        acc_list.append(test_acc)
        print('epoch:[%d/%d]\tlr = %.4f, batch_size = %d, train loss = %.3f, test acc = %.3f, elapse = %.2f sec;' \
                %(epoch+1, num_epochs, lr, batch_size, loss_acc/count, test_acc, time.time()-start))

    return loss_list, acc_list

def scored(model, x_test, y_test, batch_size, device):
    #return TP, FP, FN, TN
    model.eval().to(device)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)
    test_set = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    TP, FP, FN, TN = 0, 0, 0, 0
    for (X, y_true) in test_loader:
        output = model(X.to(device))
        output = torch.nn.functional.softmax(output, dim=1)

        y_true = y_true.to(device)
        y_pred = output.argmax(dim=1).data

        TP += torch.sum(torch.logical_and(y_true, y_pred)).item()
        FP += torch.sum(torch.logical_and(y_true, torch.logical_not(y_pred))).item()
        FN += torch.sum(torch.logical_and(torch.logical_not(y_true), y_pred)).item()
        TN += torch.sum(torch.logical_and(torch.logical_not(y_true), torch.logical_not(y_pred))).item()
    return TP, FP, FN, TN 

def pred(model, feature):
    feature = np.reshape(feature, (1, -1))
    feature = torch.from_numpy(feature)
    model.eval()
    output = model(feature)
    output = torch.nn.functional.softmax(output)
    return output.argmax(dim=1).data
    

def optimal_parameters(model_config, x_train, y_train, x_test, y_test, lr_list, batch_size_list, num_epochs, device, fig_path):
    best_lr, best_batch_size, best_acc = 0, 0, 0
    lr_loss, lr_acc = [], []
    for lr in lr_list:
        model = TextCNN(model_config)
        loss_list, acc_list = train(model, x_train, y_train, x_test, y_test, lr, 128, num_epochs, device)
        lr_loss.append(loss_list)
        lr_acc.append(acc_list)
        acc = np.array(acc_list).mean()
        if acc >= best_acc:
            best_lr = lr
            best_acc = acc
    best_acc = 0
    batch_size_loss, batch_size_acc = [], []
    for batch_size in batch_size_list:
        model = TextCNN(model_config)
        loss_list, acc_list = train(model, x_train, y_train, x_test, y_test, best_lr, batch_size, num_epochs, device)
        batch_size_loss.append(loss_list)
        batch_size_acc.append(acc_list)
        acc = np.array(acc_list).mean()
        if acc >= best_acc:
            best_batch_size = batch_size
            best_acc = acc
    
    #输出结果
    output_2(lr_loss, lr_acc, batch_size_loss, batch_size_acc, lr_list, batch_size_list, fig_path)

    return best_lr, best_batch_size