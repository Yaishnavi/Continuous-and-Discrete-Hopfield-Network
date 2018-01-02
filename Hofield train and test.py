#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 13:42:10 2017

@author: vaishnaviy
"""
import numpy as np
import sys

def discrete(x):
    return np.sign(-0.1+ np.sign(x))

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def ReLU(x):
    return x * (x > 0)

def hopfield_train(train_input, learning_rule):
        n, n_pat = train_input.shape #n_cols = number of patterns n_rows= neuron_size
        if learning_rule == "hebbian" :
            weight = np.multiply(np.matmul(train_input, train_input.T),(np.ones([n,n]) - np.identity(n)))
        elif learning_rule == "pseudo-inverse" :
            weight = np.multiply(np.matmul(train_input, np.linalg.pinv(train_input,rcond=1e-15)),(np.ones([n,n]) - np.identity(n)))
        elif learning_rule == "storkey" :
            h = np.zeros([n,n]); wmat = np.zeros([n,n]); v=0
            wmat = np.divide(np.matmul(train_input[:,v], train_input[:,v].T),n)
            for v in range(1, n_pat):
                x = train_input[:,v]
                h = np.matmul(wmat,np.matlib.repmat(x,1,n)) - np.matlib.repmat(np.multiply(np.diag(wmat),x),1,n) - np.matmul(wmat, np.diag(x))
                wmat = wmat + np.divide(np.matmul(x,x.T),n) - np.divide(np.matmul(np.diag(x),h.T),n) - np.divide(np.matmul(h,np.diag(x)),n)
            weight = wmat
        else:
            sys.exit("Invalid Learning Rule")
        return weight

    
def hopfield_retrieval(test_input, weight, n_iter, activation):
    N , n = weight.shape
    if (N!= n):
        sys.exit("Invalid Weight matrix")
    n = len(test_input) # length of test patteren ~ which should be equal to the neuron size neuron size
    if (N!= n):
        sys.exit("Length of test pattern does not match the Neuron size of the trained newtork")
        
    if  activation == "discrete":
        active = discrete
    elif activation == "sigmoid" :
        active = sigmoid
    elif activation == "tanh" :
        active = tanh
    elif activation == "ReLU" :
        active = ReLU
    else: 
        sys.exit("Invalid activation function")
        
    temp = test_input;    retrieved_output = np.zeros(n);    i = 1;    
    while i<=n_iter:
        for k in np.random.permutation(n) :
            if (weight[k,].any()!=0.0):
                temp[k] = active(np.matmul(weight[k,],temp))  
        if (temp==retrieved_output).all():
            break
        retrieved_output = temp
        i = i+1
       
    return retrieved_output

