import numpy as np

class SimpleLogisticClassifier:
    
    def __init__(self, num_features, num_classes):
        '''
        Initializes the parameters of a Logistic Classifier
        Parameters:
        num_features - int, num_classes - int
        Returns:
        None
        '''
        self.num_features = num_features
        self.num_classes = num_classes
        self.W = np.random.normal(size = (self.num_features, self.num_classes))
        self.b = np.random.normal(size = (1, self.num_classes))
        return 
        
    def predict(self, input_data, input_labels):
        '''
        Performs inference, computes crossentropy loss and accuracy
        
        Parameters:
        input_data : 2d numpy array (dtype = float, size = (num_samples, num_features))
        input_labels : 1d numpy array (dtype = int, size = (num_samples, ))
        
        Returns:
        y_preds : 1d numpy array (dtype = int, size = (num_samples, )) 
        Each entry belongs to {0, 1, ... , num_classes-1}
        
        crossentropy_loss : float, accuracy_score : float
        '''
        y_temp = np.exp(input_data @ self.W + self.b)
        y_probab = (1/np.sum(y_temp, axis = 1).reshape(-1, 1))*y_temp
        y_preds = np.argmax(y_probab, axis = 1)
        crossentropy_loss = np.mean(-1*np.log2(y_probab[np.arange(input_labels.shape[0]), y_preds]), axis = None)
        accuracy_score = np.mean((y_preds == input_labels), axis = None)
        
        return y_preds, crossentropy_loss, accuracy_score
    
    def train(self, input_data, input_labels, max_epochs = 100, eta = 1e-4, patience = 5):
        '''
        Learns the weights using simple full-batch gradient descent
        
        Parameters:
        input_data : 2d numpy array (dtype = float, size = (num_samples, num_features))
        input_labels : 1d numpy array (dtype = int, size = (num_samples, ))
        num_epochs : int, eta : float
        
        Returns:
        None        
        '''
        num_samples = input_data.shape[0]
        loss_history = []
        
        for epoch_count in range(max_epochs):
            y_temp = np.exp(input_data @ self.W + self.b)
            y_probab = (1/np.sum(y_temp, axis = 1).reshape(-1, 1))*y_temp
            y_preds = np.argmax(y_probab, axis = 1)
            crossentropy_loss = np.mean(-1*np.log2(y_probab[np.arange(num_samples), y_preds]), axis = None)
            # Early stopping
            if epoch_count >= patience and loss_history[epoch_count-patience] < crossentropy_loss:
                break
            loss_history.append(crossentropy_loss)
            # Computing gradients
            sample_labels = np.zeros((num_samples, self.num_classes), dtype = float)
            sample_labels[np.arange(num_samples), input_labels] = 1.0
            grad_preact = y_probab - sample_labels
            grad_W = (1/num_samples)*(sample_labels.T @ input_data).T
            grad_b = np.mean(grad_preact, axis = 0).reshape(1, self.num_classes)
            # Updating the parameters
            self.W = self.W - eta*grad_W
            self.b = self.b - eta*grad_b
        
        return
    