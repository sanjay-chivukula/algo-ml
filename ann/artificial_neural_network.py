import numpy as np

import activation_functions as af


class ArtificialNeuralNetwork:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        # Network
        self.num_input, self.num_hidden, self.num_output = 2, 4, 1
        self.W_xh = np.random.randn(self.num_input, self.num_hidden)
        self.b_h = np.zeros((1, self.num_hidden))
        self.W_hy = np.random.randn(self.num_hidden, self.num_output)
        self.b_y = np.zeros((1, self.num_output))

        # hyperparameters
        self.alpha = 0.01
        self.max_iterations = 3000

    def _forward_propagation(self, x):
        # layer input -> hidden
        z1 = np.dot(x, self.W_xh) + self.b_h
        a1 = af.sigmoid(z1)

        # layer hidden -> output
        z2 = np.dot(a1, self.W_hy) + self.b_y
        y_hat = af.sigmoid(z2)

        return z1, a1, z2, y_hat

    def _backward_propagation(self, x, y,  z1, a1, z2, y_hat):
        # layer output
        delta2 = np.multiply(-(y - y_hat), af.sigmoid_derivative(z2))
        dJ_dWhy = np.dot(a1.T, delta2)

        # layer hidden
        delta1 = np.dot(delta2, self.W_hy.T) * af.sigmoid_derivative(z1)
        dJ_dWxh = np.dot(x.T, delta1)

        return dJ_dWxh, dJ_dWhy

    def _loss_function(self, y, y_hat):
        J = 0.5 * sum((y - y_hat) ** 2)
        return J

    def train(self):

        loss_history = []

        for i in range(self.max_iterations):
            z1, a1, z2, y_hat = self._forward_propagation(self.x_train)
            dJ_dWxh, dJ_dWhy = self._backward_propagation(self.x_train, self.y_train, z1, a1, z2, y_hat)

            # Updating weights
            self.W_xh = self.W_xh - self.alpha * dJ_dWxh
            self.W_hy = self.W_hy - self.alpha * dJ_dWhy

            # computing loss
            loss = self._loss_function(self.y_test, y_hat)

            loss_history.append(loss)

    def test(self):
        _, _, _, y_hat = self._forward_propagation(self.x_test)
        return [(y1, y2) for y1, y2 in zip(y_hat, self.y_test)]


def test_driver():
    x = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([[1], [1], [0], [0]])

    ann_obj = ArtificialNeuralNetwork(x, y, x, y)
    ann_obj.train()
    print(ann_obj.test())


if __name__ == "__main__":
    test_driver()
