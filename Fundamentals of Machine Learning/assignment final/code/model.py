import time
import torch, numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNN

class PCA_LDA_KNN():
    def __init__(self, n_components, n_neighbors):
        self.model_pca = PCA(n_components=n_components)
        self.model_lda = LDA()
        self.model_knn = KNN(n_neighbors=n_neighbors)

    def train(self, x_train, y_train):
        data_pca = self.model_pca.fit_transform(x_train)   
        data_lda = self.model_lda.fit_transform(data_pca, y_train)
        self.model_knn.fit(data_lda, y_train)

    def acc(self, x_test, y_test):
        x_pca = self.model_pca.transform(x_test)
        x_lda = self.model_lda.transform(x_pca)
        return self.model_knn.score(x_lda, y_test)

    def pred(self):
        pass

class ForwardNeuralNetwork():
    def __init__(self, input_dim, num_hidden, output_dim):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, int(num_hidden*input_dim)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(num_hidden*input_dim), output_dim)
        ).to(self.device)

    def train(self, x_train, y_train, x_test, y_test, lr, batch_size, num_epochs):

        x_train = torch.Tensor(x_train).long()
        y_train = torch.Tensor(y_train).long()
        train_set = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)

        
        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        loss_list = []
        acc_list = []
        for epoch in range(num_epochs):
            count, loss_acc, start = 0, 0, time.time()
            # train
            for _, (feature, label) in enumerate(train_loader):
                X = Variable(feature).float().to(self.device)
                y_true = Variable(label).long().to(self.device)

                optimizer.zero_grad()

                y_hat = self.net(X)
                loss = criterion(y_hat, y_true.view(-1))

                loss.backward()
                optimizer.step()

                count += 1
                loss_acc += loss.item()
            
            # test
            test_acc = self.evaluate_accuracy(x_test, y_test, batch_size)

            # output
            loss_list.append(loss_acc/count)
            acc_list.append(test_acc)
            print('lr=%.3f, batch_size=%3d, epoch: [%d/%d]\ttrain loss: %.3f, test acc: %.3f, elapse: %.2f sec;' \
                    %(lr, batch_size, epoch+1, num_epochs, loss_acc/count, test_acc, time.time()-start))

        return loss_list, acc_list
    
    def evaluate_accuracy(self, x_test, y_test, batch_size):

        x_test = torch.Tensor(x_test)
        y_test = torch.Tensor(y_test)
        test_set = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size)

        count, acc_count = 0, 0
        for _, (X, y_true) in enumerate(test_loader):

            output = self.net(X.to(self.device))
            output = torch.nn.functional.softmax(output, dim=1)

            y_pred = output.argmax(dim=1)
            result = torch.eq(y_pred, y_true.to(self.device)).float()

            count += 1
            acc_count += torch.mean(result).item()
        return acc_count/count

    def pred(self):
        pass