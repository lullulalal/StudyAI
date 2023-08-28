import numpy as np
from NeuralNetworkTests import TestFullyConnected1
from NeuralNetworkTests import TestReLU
from NeuralNetworkTests import TestSoftMax
from NeuralNetworkTests import TestNeuralNetwork1


tfc = TestFullyConnected1()
tfc.setUp()
tfc.test_forward_size()
tfc.test_backward_size()
tfc.test_bias()

'''
relu = TestReLU()
relu.setUp()
relu.test_forward()


soft_max = TestSoftMax()
soft_max.setUp()
soft_max.test_backward_zero_loss()



nn = TestNeuralNetwork1()
nn.test_append_layer()
'''