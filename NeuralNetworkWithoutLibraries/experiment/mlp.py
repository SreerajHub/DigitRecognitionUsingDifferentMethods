#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mlp.py


import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
import src.network2 as network2
import src.mnist_loader as loader
import src.activation as act

DATA_PATH = 'C:/Users/sreer/Desktop/Fall_2018/ECE542_Neural_Networks/p1_sample/data/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Check implementation of sigmoid.')
    parser.add_argument('--gradient', action='store_true',
                        help='Gradient check')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--test', action='store_true',
                        help='Test the model')



    return parser.parse_args()

def load_data():
    train_data, valid_data, test_data = loader.load_data_wrapper(DATA_PATH)
    print('Number of training: {}'.format(len(train_data[0])))
    print('Number of validation: {}'.format(len(valid_data[0])))
    print('Number of testing: {}'.format(len(test_data[0])))
    return train_data, valid_data, test_data

def test_sigmoid():
    z = np.arange(-10, 10, 0.1)
    y = act.sigmoid(z)
    y_p = act.sigmoid_prime(z)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(z, y)
    plt.title('sigmoid')

    plt.subplot(1, 2, 2)
    plt.plot(z, y_p)
    plt.title('derivative sigmoid')
    plt.show()

def gradient_check():
    train_data, valid_data, test_data = load_data()
    model = network2.Network([784, 20, 10])
    model.gradient_check(training_data=train_data, layer_id=1, unit_id=5, weight_id=3)

def main():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    # construct the network
    model = network2.Network([784, 20,10])

    # train the network using SGD
    e_c,e_a,t_c,t_a=model.SGD(
        training_data=train_data,
        epochs=100,
        mini_batch_size=256,
        eta=1e-3,
        lmbda = 0.0,
        evaluation_data=valid_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)

    model.save('Saved_Weights')
    plt.figure(1)
    e_c_plot = np.squeeze(e_c)
    t_c_plot = np.squeeze(t_c)
    plt.plot(e_c_plot,'r',label="Validation")
    plt.plot(t_c_plot,'b',label="Training")
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    plt.title("Learning Curve: cost vs epochs")
    plt.legend(loc='upper center')
    plt.show()

    plt.figure(2)
    e_a_plot = np.squeeze(e_a)
    t_a_plot = np.squeeze(t_a)
    plt.plot(e_a_plot,'r',label="Validation")
    plt.plot(t_a_plot,'b',label="Training")
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.title("Learning Curve: accuracy vs epochs")
    plt.legend(loc='lower right')
    plt.show()




def test():

    train_data, valid_data, test_data = load_data()
    print("length of test data:",len(test_data[0]))
    net=network2.load('Saved_Weights')
    accuracy = net.accuracy(test_data)

    print("Test Accuracy",accuracy/len(test_data[0]))
    print("Test Accuracy: {} / {}".format(accuracy, len(test_data[0])))

if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        main()
    if FLAGS.gradient:
        gradient_check()
    if FLAGS.test:
        test()
