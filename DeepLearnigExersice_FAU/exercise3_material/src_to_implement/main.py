
from NeuralNetworkTests import TestConstraints
from NeuralNetworkTests import TestDropout
from NeuralNetworkTests import TestBatchNorm
from NeuralNetworkTests import TestConv
from NeuralNetworkTests import TestRNN
from NeuralNetworkTests import TestFullyConnected

rnn = TestRNN()
fcl = TestFullyConnected()
drop = TestDropout()

drop.setUp()
rnn.setUp()
fcl.setUp()

drop.test_forward_trainTime()

#fcl.test_forward_size()
#rnn.test_forward_size()
rnn.test_update()




