# Continuous and Discrete Hopfield Network
*************************************************************************
Author : Vaishnavi Yeruva
vaishnaviy2@gmail.com
*************************************************************************
# Introduction
This Python code is a simple implementation of Hopfield neural network for continious and discrete input.

# Training
run: weight = hopfield_train(train_input, learning rule)
* train_input size  = number of patterns x length of each pattern
* learning rule = "hebbian" or "pseudo-inverse" or "storkey"
* weight = output weight matrix 
        
# Testing
run: retrieved_output, i = hopfield_retrieval(test_input, weight, n_iter, activation):
* test_input = an array of single test pattern
* weight = trained weight matrix
* n_iter = choose the maximum number of iterations (very high for continuous case)
* activation = "discrete" or "sigmoid" or "tanh" or "ReLU"
* The value of input for discrete case should be {+1,-1}. The range of input for sigmoid case should be {0 to 1}
* retrieved_output = The retrieved pattern after n_iter or on reaching stability
* Asynchronous retrieval
* i = total number of iterations
        
# Requirements:
    Python 3.6
    Numpy 
