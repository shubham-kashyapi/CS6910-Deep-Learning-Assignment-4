import numpy as np

class LogisticClassifier:
    def __init__(self):
        self.flag = 0
        '''
        Initializes the Logistic Classifier
        '''
        pass

    def initialize(self):
        """
        The logistic classifier would be a
        one-vs-rest classifier
        Each W would be of size (m,1)
        Each b would be of size (1,1)
        """
        self.W_list = []
        self.b_list = []
        for _ in range(self.num_classes):
            self.W_list.append(np.random.normal(size=(self.m, 1)))
            self.b_list.append(np.random.normal(size=(1,1)))

    def get_loss(self, y_temp, y_hat):
        E = 0.5*np.linalg.norm(y_temp - y_hat)
        return E

    def get_grads(self, y_temp, y_hat):
        del_y_hat = (y_hat - y_temp)
        del_W = self.X.T@(del_y_hat*y_hat*(1-y_hat))
        del_b = np.sum(del_y_hat*y_hat*(1-y_hat))

        del_W = del_W/self.n
        del_b = del_b/self.n
        return del_W, del_b

    def get_update(self, W, b, del_W, del_b):
        W = W - self.eta*del_W
        b = b - self.eta*del_b
        return W, b

    def get_accuracy(self, cat, X, y):
        y_temp = y.copy()
        y_temp[y_temp!=cat] = -1
        y_temp[y_temp==cat] = 1
        y_temp[y_temp==-1] = 0

        W = self.W_list[cat]
        b = self.b_list[cat]

        y_hat = 1/(1+np.exp(-(X@W + b)))
        y_hat[y_hat>0.5] = 1
        y_hat[y_hat!=1] = 0

        y_temp = y_temp.reshape(-1,1)
        y_hat = y_hat.reshape(-1,1)
        # if self.flag == 0:
        #     a = y_hat==y_temp
        #     print(a.shape)
        #     print(y_hat.shape, y_temp.shape, np.sum(y_hat==y_temp))
        #     self.flag = 1

        acc = np.sum(y_hat==y_temp)/y_hat.size
        return acc

    def predict(self, X, y):
        self.probabs = np.zeros((self.n, self.num_classes))

        for cat in range(self.num_classes):
            y_temp = y.copy()
            y_temp[y_temp!=cat] = -1
            y_temp[y_temp==cat] = 1
            y_temp[y_temp==-1] = 0

            W = self.W_list[cat]
            b = self.b_list[cat]

            y_hat = 1/(1+np.exp(-(X@W + b)))

            self.probabs[:, cat] = y_hat.reshape(-1,)

        y_pred = np.argmax(self.probabs, axis=1)
        acc = np.sum(y_pred==y)/y.size
        return y_pred, acc
    
    def fit(self, X, y, epochs=100, eta=1e-1, patience=5):
        '''
        Learns the weights using simple full-batch gradient descent
        
        Parameters:
        X : 2d numpy array (dtype = float, size = (n, m))
        n is the number of samples and 
        m is the number of features
        input_labels : 1d numpy array (dtype = int, size = (n, ))
        num_epochs : int, eta : float
        
        Returns:
        None        
        '''
        self.n = X.shape[0]
        self.m = X.shape[1]
        self.num_classes = np.unique(y).size

        self.initialize()
        self.X = X
        self.y = y
        self.eta = eta
        self.epochs = epochs

        self.loss_history = {}
        self.acc_history = {}

        # Class-wise training
        # cat stands for individual category
        for cat in range(self.num_classes):
            # Preprocess the y for 
            # one-vs-rest classification
            y_temp = y.copy()
            y_temp[y_temp!=cat] = -1
            y_temp[y_temp==cat] = 1
            y_temp[y_temp==-1] = 0
            y_temp = y_temp.reshape(-1,1)

            W = self.W_list[cat]
            b = self.b_list[cat]

            self.loss_history[cat] = []
            self.acc_history[cat] = []

            for _ in range(self.epochs):
                # Loss-function is MSE
                y_hat = 1/(1+np.exp(-(X@W + b)))
                E = self.get_loss(y_temp, y_hat)
                del_W, del_b = self.get_grads(y_temp, y_hat)

                W, b = self.get_update(W, b, del_W, del_b)

                self.W_list[cat] = W
                self.b_list[cat] = b
                
                acc = self.get_accuracy(cat, X, y)
                self.loss_history[cat].append(E)
                self.acc_history[cat].append(acc)
                
        