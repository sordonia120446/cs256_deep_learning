"""
For this homework you will create a Python implementation of a Perceptron together with an implementation of a couple Perceptron training algorithms. You will conduct some experiments to see if they confirm or deny what we learned about the Perceptron's ability to PAC-learn various functions.

:author: Sam O <samuel.ordonia@sjsu.edu>
"""

import random
from math import tanh

import numpy as np


# test input
data = np.array([ [0,0,1], 
    [0,1,1],
    [1,0,1],
    [1,1,1] ]
)
#y = np.array( [ [0],[1],[1],[0] ] )
y = np.array( [ [0],[1],[1],[1] ] )

# seeding to help debug
random.seed(1)

# num steps
n = 10000

# weights
#weights = 2*np.random.random((4, 1)) - 1  # 4x1 matrix of weights with bias of 1


def get_act_func(func_name='tanh'):

    if func_name == 'threshold':
        return lambda x, theta: 0 if x < theta else 1
    elif func_name == 'tanh':
        return lambda x, theta: (0.5) + (0.5)*tanh((x - theta)*0.5)
    elif func_name == 'relu':
        return lambda x, theta: max(0, x - theta)
    elif func_name == 'test':
        return lambda x, theta: x + theta
    else:
        raise Exception('not a recognized activation function')

def nonlin(x, deriv=False):
    """
        Sigmoid function
    """
    if (deriv == True):
            return( x*(1-x) )

    return (1/( 1 + np.exp(-x) ) )

def perceptron_fnc(weights, theta, error, x):
    x = np.array([x])
    for e in error:
        if e > 0:
            weights -= x.T
            #theta += 1
        elif e < 0:
            weights += x.T
            #theta -= 1
        return weights, theta


def winnow_fnc(weights, theta, error):
    alpha = 2
    if error > 0:
        weights *= alpha**(-1)
    elif error < 0:
        weights *= alpha**(1)
    return weights, theta


def get_training_function(func_name):
    if func_name == 'perceptron':
        return perceptron_fnc
    elif func_name == 'winnow':
        return winnow_fnc
    else:
        raise Exception('not a recognized training function')


# TODO read in and parse ground file


# TODO determine dist and generate training/test sets


# TODO train and test model


def run_perceptron(args):
    act_func = get_act_func(args.activation)
    #act_func = nonlin
    training_func = (get_training_function(args.training_alg))
    theta = 0.0
    weights = 2*np.random.random((3, 1)) - 1  # 3x1 matrix of weights with bias of 1
    for i in xrange(100):
        for ind, x in enumerate(data):
            scalar_thing = np.dot(x, weights)
            y_pred = act_func(np.dot(x, weights), theta)
            error = y[ind] - y_pred
            weights, theta = training_func(weights, theta, error, x)

        if i % 10 == 0:
            print 'scalar dot product {}'.format(scalar_thing)
            print 'theta = {}'.format(theta)
            print 'weights = {}'.format(weights)

    print 'Truth {}'.format(y[ind])
    print 'prediction {}'.format(y_pred)
    print 'final error {}'.format(error)

