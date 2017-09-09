"""
For this homework you will create a Python implementation of a Perceptron together with an implementation of a couple Perceptron training algorithms. You will conduct some experiments to see if they confirm or deny what we learned about the Perceptron's ability to PAC-learn various functions.

:author: Sam O <samuel.ordonia@sjsu.edu>
"""

from math import tanh


def get_activation_function(func_name):
    if func_name == 'threshold':
        return lambda x, theta: 0 if x < theta else 1
    elif func_name == 'tanh':
        return lambda x, theta: (1/2) + (1/2)*tanh((x - theta)/2)
    elif func_name == 'relu':
        return lambda x, theta: max(0, x - theta)
    else:
        print 'not a recognized activation function'
        raise


# TODO choose training algo


# TODO read in and parse ground file


# TODO determine dist and generate training/test sets


# TODO train and test model


def run_perceptron(args):
    act_func = get_activation_function(args.activation)
    print act_func

