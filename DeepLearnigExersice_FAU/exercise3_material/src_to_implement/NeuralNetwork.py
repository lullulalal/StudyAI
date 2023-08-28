import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        weight_decay = 0

        for layer in self.layers:
            layer.testing_phase = False
            input_tensor = layer.forward(input_tensor)
            if layer.trainable is True and layer.optimizer is not None and layer.optimizer.regularizer is not None:
                weight_decay += layer.weights_norm

        return self.loss_layer.forward(input_tensor, self.label_tensor) + weight_decay

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)

        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable is True:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)

        self.layers.append(layer)

    def train(self, iterations):
        for idx in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        for layer in self.layers:
            layer.testing_phase = True
            input_tensor = layer.forward(input_tensor)

        return input_tensor