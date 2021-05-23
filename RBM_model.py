import numpy as np

class RBM:
    def __init__(self, num_visible, num_hidden):
        '''
        Initializes the parameters of a RBM
        Parameters:
        num_visible - int, num_hidden - int
        Returns:
        None
        '''
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        #######################################
        # Random initialization of parameters
        #######################################
        self.W = np.random.normal(size = (self.num_hidden, self.num_visible))
        self.b = np.random.normal(size = (1, self.num_visible))
        self.c = np.random.normal(size = (self.num_hidden, 1))
        return
    
    def check_data_format(self, input_data):
        '''
        Checks the format of visible variables
        
        Parameters:
        input_data - 2d numpy array (dtype = float, size = (num_samples, num_visible))
        Should contain only 0's and 1's
        
        Returns:
        None
        '''
        if input_data.ndim != 2:
            raise ValueError("Number of dimensions of the input array should be 2")
        if input_data.shape[1] != self.num_visible:
            raise ValueError("The input array should have {} columns".format(self.num_visible))
        if not(((input_data == 0.0) | (input_data == 1.0)).all()):
            raise ValueError("The input array should contain only 0's and 1's")
        return
    
    def get_hidden_rep(self, input_data):
        '''
        Given the visible variables, computes the hidden representation using sampling.
        
        Parameters:
        input_data - 2d numpy array (dtype = float, size = (num_samples, num_visible))
        Should contain only 0's and 1's
        
        Returns:
        hidden_rep - 2d numpy array (dtype = float, size = (num_samples, num_hidden))
        Contains only 0's and 1's
        '''
        ###################################
        # Checking the format of the data
        ###################################
        input_data = input_data.astype(float)
        self.check_data_format(input_data)
        
        ###################################
        # Sampling the hidden variables
        ###################################
        num_samples = input_data.shape[0]
        hidden_probabs = (1/(1+np.exp(self.W @ input_data.T + self.c))).T
        # print(hidden_probabs)
        random_vals = np.random.uniform(low = 0.0, high = 1.0, size = (num_samples, self.num_hidden))
        hidden_rep = (random_vals > hidden_probabs).astype(float)
        return hidden_rep        
        
        
        
    def train_Gibbs_Sampling(self, input_data, k, r, eta):
        '''
        Performs one epoch of training using Block Gibbs Sampling. 
        Weights are updated after processing each training sample.
        
        Parameters:
        data - 2d numpy array (dtype = float, size = (num_samples, num_visible))
        Should contain only 0's and 1's
        k - Number of steps to be run for convergence (int)
        r - Number of samples to be drawn after convergence (int)
        eta - Learning rate (float)
        
        Returns:
        None
        '''
        ###################################
        # Checking the format of the data
        ###################################
        input_data = input_data.astype(float)
        self.check_data_format(input_data)
        
        ###################################    
        # Block Gibbs Sampling
        ###################################
        for row in range(input_data.shape[0]):
            W_update_sum = np.zeros((self.num_hidden, self.num_visible), dtype = float)
            b_update_sum = np.zeros((1, self.num_visible), dtype = float)
            c_update_sum = np.zeros((self.num_hidden, 1), dtype = float)
            # Randomly initializing the visible variables
            visible_vars = np.random.randint(low = 0, high = 2, size = (self.num_visible, 1)).astype(float)     
            
            for step_num in range(k+r):
                # Sampling the hidden variables given the visible variables
                hidden_probs = 1/(1 + np.exp(self.W @ visible_vars + self.c)) # P(hidden var = 0 | visible vars)
                random_vals = np.random.uniform(low = 0.0, high = 1.0, size = (self.num_hidden, 1))
                hidden_vars = np.array((random_vals > hidden_probs), dtype = float).reshape(self.num_hidden, 1)
                # Sampling the visible variables given the hidden variables
                # print(hidden_vars.shape, self.W.shape, self.b.shape, self.c.shape)
                visible_probs = 1/(1 + np.exp(hidden_vars.T @ self.W + self.b)) # P(visible var = 0 | hidden vars)
                random_vals = np.random.uniform(low = 0.0, high = 1.0, size = (1, self.num_visible))
                visible_vars = np.array((random_vals > visible_probs), dtype = float).reshape(self.num_visible, 1)
                # Incrementing the updates only after step_num >= k i.e. the Markov chain has converged
                if step_num >= k:
                    W_update_sum += ((1/(1 + np.exp(-(self.W @ visible_vars + self.c)))) @ visible_vars.T)
                    b_update_sum += visible_vars.T
                    c_update_sum += (1/(1 + np.exp(-(self.W @ visible_vars + self.c))))
                
            # Updating the weights
            curr_visible = input_data[row, :].reshape(self.num_visible, 1) 
            curr_sigmoid = (1/(1 + np.exp(-(self.W @ curr_visible + self.c))))
            self.W += eta*((curr_sigmoid @ curr_visible.T) - (1.0/r)*W_update_sum)
            self.b += eta*(curr_visible.T - (1.0/r)*b_update_sum)
            self.c += eta*(curr_sigmoid - (1.0/r)*c_update_sum)
            
        return
    
    
    def train_Contrastive_Divergence(self, input_data):
        pass                     