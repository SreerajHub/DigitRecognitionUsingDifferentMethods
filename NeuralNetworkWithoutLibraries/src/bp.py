#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py

import numpy as np
from src.activation import sigmoid, sigmoid_prime

def backprop(x, y, biases, weights, cost, num_layers):
    """ function of backpropagation
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of all biases and weights.

        Args:
            x, y: input image x and label y
            biases, weights (list): list of biases and weights of entire network
            cost (CrossEntropyCost): object of cost computation
            num_layers (int): number of layers of the network

        Returns:
            (nabla_b, nabla_w): tuple containing the gradient for all the biases
                and weights. nabla_b and nabla_w should be the same shape as 
                input biases and weights
    """
    # initial zero list for store gradient of biases and weights
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    """"
    print("x.shape",x.shape)
    print("y",y.shape)
    print("b",biases[1].shape)
    print("l_w",len(weights))
    print("w",weights[1].shape)
    """

    ### Implement here
    # feedforward
    # Here you need to store all the activations of all the units
    # by feedforward pass
    ###
    index=0
    activations=[np.zeros(b.shape) for b in biases]
    input =x
    for b,w in zip(biases,weights):
        activations[index]=sigmoid(np.dot(w,x)+b)
        x=activations[index]
        index +=1


    # compute the gradient of error respect to output
    # activations[-1] is the list of activations of the output layer
    delta = (cost).delta(activations[-1], y)
    gradlayer =delta/(activations[-1]*(1-activations[-1]))

    ### Implement here
    # backward pass
    # Here you need to implement the backward pass to compute the
    # gradient for each weight and bias
    ###
    for y in range(0,len(activations)):
        gradactivation=np.multiply(gradlayer,sigmoid_prime(np.log((activations[-1-y])/(1-activations[-1-y]))))
        nabla_b[-1-y]=gradactivation
        if (y == (len(activations) - 1)):
            activation = input
        else:
            activation=activations[-2-y]
        nabla_w[-1-y]=np.dot(gradactivation,activation.transpose())
        gradlayer=np.dot(weights[-1-y].transpose(),gradactivation)

    return (nabla_b, nabla_w)

