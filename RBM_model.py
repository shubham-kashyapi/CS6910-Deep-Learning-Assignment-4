import numpy as np
from tqdm import tqdm

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
        input_data - 2d numpy array (dtype=float, size = (num_samples, num_visible))
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
        input_data - 2d numpy array (dtype=float, size=(num_samples, num_visible))
        Should contain only 0's and 1's
        
        Returns:
        hidden_rep - 2d numpy array (dtype=float, size=(num_samples, num_hidden))
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
        random_vals = np.random.uniform(low=0.0, high=1.0, size=(num_samples, self.num_hidden))
        hidden_rep = (random_vals > hidden_probabs).astype(float)
        return hidden_rep
    
    def get_visible_rep(self, hidden_data):
        '''
        Given the hidden variables, computes the visible representation using sampling.
        
        Parameters:
        hidden_data - 2d numpy array (dtype=float, size=(num_samples, num_hidden))
        Should contain only 0's and 1's
        
        Returns:
        visible_rep - 2d numpy array (dtype=float, size=(num_samples, num_visible))
        Contains only 0's and 1's
        '''
        ###################################
        # Checking the format of the data
        ###################################
        hidden_data = hidden_data.astype(float)
        if hidden_data.shape[1] != self.num_hidden:
            raise ValueError("The input array should have {} columns".format(self.num_hidden))
        self.W = self.W.reshape(self.num_hidden, self.num_visible)
        self.b = self.b.reshape(1, self.num_visible)
        self.c = self.c.reshape(self.num_hidden, 1)
        
        ###################################
        # Sampling the hidden variables
        ###################################
        num_samples = hidden_data.shape[0]        
        visible_probabs = 1/(1 + np.exp(hidden_data @ self.W + self.b)) # P(visible var = 0 | hidden vars)
        random_vals = np.random.uniform(low = 0.0, high = 1.0, size = (num_samples, self.num_visible))
        visible_rep = (random_vals > visible_probabs).astype(float)
        return visible_rep
    
        
    def train_Gibbs_Sampling(self, input_data, k, r, eta=1e-4):
        '''
        Performs one epoch of training using Block Gibbs Sampling. 
        Weights are updated after processing each training sample.
        
        Parameters:
        data - 2d numpy array (dtype = float, size=(num_samples, num_visible))
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
        
    def sigmoid(self, x):
        val = 1/(1+np.exp(-x))
        return val

    def sample_h_vec(self, v):
        # print(self.W.shape, v.shape, self.c.shape)
        val = self.sigmoid((self.W @ v).reshape(-1,1) + self.c)
        flag = np.random.uniform(size=val.size).reshape(-1,1)
        # print(val.shape, flag.shape)
        return (flag < val).astype("float")

    def sample_v_vec(self, h):
        # print(self.W.T.shape, h.shape, self.b.shape)
        val = self.sigmoid((self.W.T @ h).reshape(-1,1) + self.b)
        flag = np.random.uniform(size=val.size).reshape(-1,1)
        # print(val.shape, flag.shape)
        return (flag < val).astype("float")

    def sample_h(self, W, c, v):
        # print(W.shape, v.shape, self.c.shape)
        val = self.sigmoid((W @ v) + c)
        flag = np.random.uniform(size=val.shape)
        # print(val.shape, flag.shape)
        return (flag < val).astype("float")

    def sample_v(self, W, b, h):
        # print(W.T.shape, h.shape, b.shape)
        val = self.sigmoid((W.T @ h) + b)
        flag = np.random.uniform(size=val.shape)
        # print(val.shape, flag.shape)
        return (flag < val).astype("float")

    def kstep_cd(self, v):
        for _ in range(self.k):
            h = self.sample_h_vec(v)
            # print("h.shape:", h.shape)
            v = self.sample_v_vec(h)
        return v

    def get_grads(self, curr, recons):
        # print("curr:", curr.shape, "recons:", recons.shape, "W:", self.W.shape, "c:", self.c.shape)
        term1 = (self.sigmoid((self.W@curr).reshape(-1,1) + self.c))@ curr.T
        term2 = (self.sigmoid((self.W@recons).reshape(-1,1) + self.c))@ curr.T
        del_W = term1 - term2
        # print("del_W:", del_W.shape)

        del_b = curr - recons
        # print("b:", self.b.shape)
        # print("del_b:", del_b.shape)
        term1 = self.sigmoid((self.W@curr).reshape(-1,1) + self.c)
        term2 = self.sigmoid((self.W@recons).reshape(-1,1) + self.c)
        # print(self.c.shape)
        # print(term1.shape, term2.shape)
        del_c = term1 - term2

        # print("del_b shape:", del_b.shape)
        # print("del_c shape:", del_c.shape)
        return del_W, del_b, del_c

    def get_loss(self, curr):
        h = self.sample_h_vec(curr)
        v = self.sample_v_vec(h)
        loss = np.sqrt(np.mean((v-curr)**2))
        return loss
    
    def train_contrastive_divergence(self, input_data, eta):
        '''
        Performs one epoch of training using Contrastive Divergence. 
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
        self.b = self.b.reshape(-1,1)
        
        self.param_SGD = {"W":[], "b":[], "c":[]}
        self.param_hist = {"W":[], "b":[], "c":[]}
        self.overall_loss = []
        for epoch in tqdm(range(self.epochs)):
            loss_list = []

            # SGD
            for row in range(input_data.shape[0]):
                v0 = input_data[row, :].copy()
                v = input_data[row, :].copy()
                # print("v.shape:", v.shape)
                # print("hidden_layer size:", self.num_hidden)
                
                recons = self.kstep_cd(v)

                # Updating the weights
                curr = input_data[row, :].reshape(-1, 1) 
                del_W, del_b, del_c = self.get_grads(curr, recons)

                self.W += eta*del_W
                self.b += eta*del_b
                self.c += eta*del_c

                loss = self.get_loss(curr)
                loss_list.append(loss)

                if epoch == self.epochs-1:
                    self.param_SGD["W"].append(self.W.copy())
                    self.param_SGD["b"].append(self.b.copy())
                    self.param_SGD["c"].append(self.c.copy())


            self.overall_loss.append(np.mean(loss_list))
            self.param_hist["W"].append(self.W.copy())
            self.param_hist["b"].append(self.b.copy())
            self.param_hist["c"].append(self.c.copy())

    def train(self, train_type, input_data, k=None, epochs=None, r=None, eta=None):
        self.k = k
        self.r = r
        self.eta = eta
        self.epochs = epochs

        if train_type=="CD":
            self.train_contrastive_divergence(input_data, eta)
        if train_type=="BGS":
            self.train_Gibbs_Sampling(input_data, k, r, eta)
