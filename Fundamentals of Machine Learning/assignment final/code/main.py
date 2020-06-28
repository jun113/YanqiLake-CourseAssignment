from utils import mkdir, load_dataset, output_assignment_1, output_assignment_2
from model import PCA_LDA_KNN, ForwardNeuralNetwork

class config():
    def __init__(self, save_result):
        self.save_path = mkdir() if save_result else None
        self.dataset = ['Letter Recognition', 'MNIST']
    
    def config_1(self):
        self.PCA_components = {'MNIST': [50, 100, 200, 300, 400], \
                  'Letter Recognition': [3, 5, 7, 9, 11]}
        self.KNN_neighbors = [1, 3]

    def config_2(self):
        self.num_labels = {'MNIST': 10, 'Letter Recognition': 26}
        self.feature_dim = {'MNIST': 784, 'Letter Recognition': 16}
        self.lr = [0.001, 0.01, 0.03, 0.05, 0.1]
        self.batch_size = [32, 64, 128, 256, 512]
        self.num_epochs = 150
        self.num_hidden = [1, 1.5, 2, 2.5, 3]

    def assignment_1(self):
        '''
        PCA+LDA+KNN parameters:
        PCA n_components, 
            Letter Recognition: 3, 5, 7, 9, 11;
            MNIST: 50, 100, 200, 300, 400.
        K-NN n_neighbors: 1, 3.
        '''
        print('start assignment_1: PCA+LDA+KNN')
        self.config_1()

        for dataset_name in self.dataset:
            x_train, y_train, x_test, y_test = load_dataset(dataset_name)
        
            result = []

            for n_components in self.PCA_components[dataset_name]:
                acc_list = []
                for n_neighbors in self.KNN_neighbors:
                    model = PCA_LDA_KNN(n_components, n_neighbors)
                    model.train(x_train, y_train)
                    acc = model.acc(x_test, y_test)
                    acc_list.append(acc)

                result.append(acc_list)
            
            output_assignment_1(result, dataset_name, self.PCA_components[dataset_name], self.KNN_neighbors, save_path=self.save_path)     

    def assignment_2(self):
        self.config_2()

        print('start assignment_2: Feed Forward Neural Network')

        for dataset_name in self.dataset:
            x_train, y_train, x_test, y_test = load_dataset(dataset_name)

            acc_hidden, loss_hidden = [], []
            print('num_hidden...')
            for num_hidden in self.num_hidden:
                model = ForwardNeuralNetwork(self.feature_dim[dataset_name], num_hidden, self.num_labels[dataset_name])

                loss_list, acc_list = model.train(x_train, y_train, x_test, y_test, \
                                lr=1e-3, batch_size=128, num_epochs=self.num_epochs)
                acc_hidden.append(acc_list)
                loss_hidden.append(loss_list)

            loss_lr, acc_lr  = [], []
            print('lr...')
            for lr in self.lr:
                model = ForwardNeuralNetwork(self.feature_dim[dataset_name], 2, self.num_labels[dataset_name])
                
                loss_list, acc_list = model.train(x_train, y_train, x_test, y_test, \
                                lr=lr, batch_size=128, num_epochs=self.num_epochs)
                acc_lr.append(acc_list)
                loss_lr.append(loss_list)
            
            loss_batch_size, acc_batch_size = [], []
            print('batch_size...')
            for batch_size in self.batch_size:
                model = ForwardNeuralNetwork(self.feature_dim[dataset_name], 2, self.num_labels[dataset_name])
                
                loss_list, acc_list = model.train(x_train, y_train, x_test, y_test, \
                                lr=1e-3, batch_size=batch_size, num_epochs=self.num_epochs)
                acc_batch_size.append(acc_list)
                loss_batch_size.append(loss_list)

            output_assignment_2(acc_hidden, loss_hidden, acc_lr, loss_lr, acc_batch_size, loss_batch_size, \
                2, 1e-3, 128, \
                dataset_name, self.feature_dim[dataset_name], self.num_labels[dataset_name], \
                    self.num_hidden, self.lr, self.batch_size, self.save_path)
if __name__ == '__main__':
    run = config(save_result=True)
    run.assignment_1()
    run.assignment_2()