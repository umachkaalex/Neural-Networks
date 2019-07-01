import numpy as np
import pandas as pd
import time
from tqdm import tqdm
np.set_printoptions(suppress=True)


class VectorNN:
    def __init__(self, model_params):
        # internals
        self.X: np.array
        self.Y: np.array
        self.weights: np.array
        self.accuracies: np.array
        self.l_act = []
        self.classes = []
        self.thetas = []
        self.thetas_deriv = []
        self.activations = []
        self.rows = int
        self.J_hist = []
        self.predictions = []
        self.pred = []
        self.accuracy = float
        # properties
        self.img_fldr = model_params['dataset']
        self.lambda_ = model_params['lambda']
        self.l_rate = model_params['learning_rate']
        self.iterations = model_params['iterations']
        self.target = model_params['target_column']
        self.hiddens = model_params['hidden_layers']
        self.train_split = 1 - model_params['test_split']
        self.print_split = model_params['print_split']
        self.func_type = model_params['func_type']

    def run(self):
        self.prepare_datasets()
        self.create_layers()
        self.init_weigths()
        print(self.iterations)
        for i in tqdm(range(self.iterations)):
            if int(i / self.print_split) - i / self.print_split == 0 or \
                    i == self.iterations-1:
                time.sleep(0.1)
                print('\nLast 10 costs: ' + str(i) + ': ' + str(np.round(self.J_hist[-10:], 4)))
                speed = np.asarray(self.J_hist[-10:]) - np.asarray(self.J_hist[-11:-1])
                print('Speed of changes: ' + str(np.sum(speed)))
                if self.train_split != 1:
                    self.forward_propogation(i, train=False)
            self.forward_propogation(i)
            self.compute_cost()
            self.backpropogation()
            for i in range(len(self.thetas)):
                self.thetas[i] = self.thetas[i] - self.l_rate*self.thetas_deriv[i]

        return self.thetas

    def activ_func(self, z):
        if self.func_type == 'sigmoid':
            act_func = 1 / (1 + np.exp(-z))
        if self.func_type == 'tanh':
            act_func = (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
        return act_func

    def prepare_datasets(self):
        self.X = pd.read_csv(self.img_fldr)
        self.X = self.X.sample(frac=1).reset_index(drop=True)
        y_base = self.X[self.target].values
        self.classes = list(set(y_base))
        self.Y = np.zeros([len(y_base), len(self.classes)])
        for i in range(len(self.classes)):
            cur_class = self.classes[i]
            self.Y[:, i] = (y_base == cur_class) * 1
        self.X.drop(columns=[self.target], inplace=True)
        self.X = self.X.values
        self.rows = self.X.shape[0]

    def create_layers(self):
        self.layers = [self.X.shape[1], self.Y.shape[1]]
        for i in range(len(self.hiddens)):
            self.layers.insert(i + 1, self.hiddens[i])
        self.X = np.concatenate([np.ones((self.X.shape[0], 1)), self.X], axis=1)

    def init_weigths(self):
        for j in range(len(self.layers)-1):
            self.thetas.append(np.random.uniform(size=(self.layers[j+1], self.layers[j]+1))/1000)
            self.thetas_deriv.append(np.zeros(self.thetas[-1].shape))

    def forward_propogation(self, i, train=True):
        if train:
            X = self.X[:int(len(self.X)*self.train_split), :]
            y = self.Y[:int(len(self.X)*self.train_split), :]
        else:
            X = self.X[int(len(self.X) * self.train_split):, :]
            y = self.Y[int(len(self.X) * self.train_split):, :]

        self.l_act = []
        zs = np.matmul(X, self.thetas[0].T)
        a1 = self.activ_func(zs)
        self.l_act.append(np.concatenate([np.ones((a1.shape[0], 1)), a1], axis=1))
        if len(self.layers)>3:
            for l in range(1, len(self.layers)-1):
                cur_z = np.matmul(self.l_act[-1], self.thetas[l].T)
                cur_a = self.activ_func(cur_z)
                if l < len(self.layers)-2:
                    self.l_act.append(np.concatenate([np.ones((cur_a.shape[0], 1)), cur_a], axis=1))
        zf = np.dot(self.l_act[-1], self.thetas[-1].T)
        af = self.activ_func(zf)
        self.l_act.append(af)
        self.pred = np.round(af)
        self.accuracy = self.my_accuracy_score(self.pred, y)
        if int(i / self.print_split) - i / self.print_split == 0 and train:
            print('Current train accuracy: ' + str(self.accuracy))
        if not train and int(i / self.print_split) - i / self.print_split == 0:
            print('Current test accuracy: ' + str(self.accuracy))
        if train and i == self.iterations - 1:
            print('Final train accuracy: ' + str(self.accuracy))
        if not train and i == self.iterations - 1:
            print('********')
            print('Final test accuracy: ' + str(self.accuracy))


    def compute_cost(self):
        y = self.Y[:int(len(self.X) * self.train_split), :]
        Jb = sum(-1/self.rows*sum((y*np.log(self.l_act[-1]) + (1 - y)*np.log(1-self.l_act[-1]))))
        Jt = 0
        for theta in self.thetas:
            Jt += sum(sum(self.lambda_/(2*self.rows)*np.power(theta[:, 1:theta.shape[1]], 2)))
        J = Jb + Jt
        self.J_hist.append(J)

    def backpropogation(self):
        X = self.X[:int(len(self.X) * self.train_split), :]
        y = self.Y[:int(len(self.X) * self.train_split), :]
        deltas = []
        lambdas = []
        deltas.append(self.l_act[-1] - y)
        lambdas.append(self.lambda_ / self.rows * np.concatenate([np.zeros([self.thetas[-1].shape[0], 1]),
                                                                  self.thetas[-1][:, 1:]], axis=1))
        self.thetas_deriv[-1] = 1 / self.rows * np.matmul(deltas[-1].T, self.l_act[-2]) + lambdas[-1]
        for l in range(2, len(self.layers) - 1):
            ind = len(self.layers) - l
            deltas.append(np.multiply(np.matmul(deltas[-1], self.thetas[ind]),
                                      self.l_act[ind - 1]*(1 - self.l_act[ind - 1]))[:, 1:])
            lambdas.append(self.lambda_ / self.rows * np.concatenate([np.zeros([self.thetas[ind-1].shape[0], 1]),
                                                              self.thetas[ind-1][:, 1:]], axis=1))
            self.thetas_deriv[ind-1] = 1 / self.rows * np.matmul(deltas[-1].T, self.l_act[ind-2]) + lambdas[-1]

        deltas.append(np.multiply(np.matmul(deltas[-1], self.thetas[1]), self.l_act[0] * (1 - self.l_act[0]))[:, 1:])
        lambdas.append(self.lambda_ / self.rows * np.concatenate([np.zeros([self.thetas[0].shape[0], 1]),
                                                                  self.thetas[0][:, 1:]], axis=1))
        self.thetas_deriv[0] = 1 / self.rows * np.matmul(deltas[-1].T, X) + lambdas[-1]

    def my_accuracy_score(self, y_pred, y):
        return np.mean((y_pred == y) * 1)
