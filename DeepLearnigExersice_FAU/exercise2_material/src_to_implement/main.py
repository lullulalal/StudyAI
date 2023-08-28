import numpy as np
import Layers.Conv
import Layers.Pooling

from NeuralNetworkTests import TestFlatten
from NeuralNetworkTests import TestReLU
from NeuralNetworkTests import TestSoftMax
from NeuralNetworkTests import TestConv
from NeuralNetworkTests import TestPooling
'''
conv = Layers.Conv.Conv((2, 2), (1, 2, 2), 1)
input_tensor = np.random.uniform(0.0, 1.0, (1, 1, 3, 4))
output_tensor = conv.forward(input_tensor)


pooling = TestPooling()
pooling.setUp()
pooling.test_gradient_stride()


conv = TestConv()
conv.setUp()
conv.test_backward_size()

'''

pooling = Layers.Pooling.Pooling((1,1), (3,3))
i = np.random.uniform(0.0, 1.0, (1, 2, 7, 7))
o = pooling.forward(i)
e = pooling.backward(o)
