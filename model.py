# AI for Breakout

# Importing the librairies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initializing and setting the variance of a tensor of weights
# Set the standard deviation for initializing 2 fc layers's (actor & critic) weight
# Samll deviation for actor, bigger for critic
def normalized_columns_initializer(weights, std = 1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out)) # var(out) = std^2
    return out

# Initializing the weights of the neural network in an optimal way for the learning
def weights_init(m):
    # m is NN object
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4]) # dim1 * dim2 * dim3
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0] # dim0 * dim2 * dim3
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
        m.weight.data.uniform_(-w_bound, w_bound) # generating some random weights of order inversely proportional to the size of the tensor of weights
        m.bias.data.fill_(0) # initializing all the bias with zeros
    
    # For fc layer    
    elif classname.find('Linear') != -1: # if the connection is a full connection
        weight_shape = list(m.weight.data.size()) # list containing the shape of the weights in the object "m"
        fan_in = weight_shape[1] # dim1
        fan_out = weight_shape[0] # dim0
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    
# Making the A3C brain
class ActorCritic(nn.Module):
    def __init__(self, num_input, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_input, 32, 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        # LSTM for the memory
        # Since the images are sequence of 4 images
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)
        # Same encoded state vector for Actor and Critic
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1) # V(s)
        self.actor_linear = nn.Linear(256, num_outputs) # Q(s, a)
        self.apply(weights_init) # initilizing the weights of the model with random weights
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01) # setting the standard deviation of the actor tensor of weights to 0.01
        self.actor_linear.bias.data.fill_(0) # initializing the actor bias with zeros
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0) # setting the standard deviation of the critic tensor of weights to 0.01
        self.critic_linear.bias.data.fill_(0) # initializing the critic bias with zeros
        self.lstm.bias_ih.data.fill_(0) # initializing the lstm bias with zeros
        self.lstm.bias_hh.data.fill_(0) # initializing the lstm bias with zeros
        self.train() # setting the module in "train" mode to activate the dropouts and batchnorms
    
    def forward(self, inputs):
        inputs, (hx, cx) = inputs # getting separately the input images to the tuple (hidden states, cell states)
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 32 * 3 * 3) # flattening the last convolutional layer into this 1D vector x
        hx, cx = self.lstm(x, (hx, cx)) # the LSTM takes as input x and the old hidden & cell states and ouputs the new hidden & cell states
        x = hx # getting the useful output, which are the hidden states (principle of the LSTM)
        return self.critic_linear(x), self.actor_linear(x), (hx, cx) # returning the output of the critic (V(S)), the output of the actor (Q(S,A)), and the new hidden & cell states ((hx, cx))
        
