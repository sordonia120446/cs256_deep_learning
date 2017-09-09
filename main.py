"""
Homework 1:  Perceptron

:author: Sam O <samuel.ordonia@sjsu.edu>
"""

import argparse

from scripts.perceptron import run_perceptron


"""CLARGS"""
parser = argparse.ArgumentParser(
    description='CS 256 Homework 1',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='For further questions, please consult the README.'
)

# Add CLARGS
parser.add_argument(
    '-p',
    '--perceptron',
    action='store_true',
    help='Calls in perceptron stuff'
)
parser.add_argument(
    'activation',
    choices=['threshold', 'tanh', 'relu'],
    help='Input activation function: `threshold`, `tanh`, or `relu`'
)
parser.add_argument(
    'training_alg',
    choices=['perceptron', 'winnow'],
    help='For updating weights, choose `perceptron` or `winnow`.'
)
parser.add_argument(
    'ground_file',
    help='Specify file location for ground function f(x) that generates\
    ground truth.'
)


if __name__ == '__main__':
    args = parser.parse_args()

    if args.perceptron:
        run_perceptron(args)

