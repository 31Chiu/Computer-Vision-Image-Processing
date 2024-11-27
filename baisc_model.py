import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

def mse_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

class NeuralNetwork:
    def __init__(self):
        self.w1 = np.random.rand()
        self.w2 = np.random.rand()
        self.w3 = np.random.rand()
        self.w4 = np.random.rand()
        self.w5 = np.random.rand()
        self.w6 = np.random.rand()

    def feedforward(self, X):
        h1 = sigmoid(X[0] * self.w1 + X[1] * self.w3)
        h2 = sigmoid(X[0] * self.w2 + X[1] * self.w4)

        o1 = sigmoid(h1 * self.w5 + h2 * self.w6)

        return h1, h2, o1
    
    def backpropagation(self, X, y, h1, h2, o1, learning_rate=0.1):
        dC_dw5 = -2 * (y - o1) * sigmoid_derivative(o1) * h1
        dC_dw6 = -2 * (y - o1) * sigmoid_derivative(o1) * h2
        dC_dw1 = -2 * (y - o1) * sigmoid_derivative(o1) * self.w5 * sigmoid_derivative(h1) * X[0]
        dC_dw2 = -2 * (y - o1) * sigmoid_derivative(o1) * self.w6 * sigmoid_derivative(h2) * X[0]
        dC_dw3 = -2 * (y - o1) * sigmoid_derivative(o1) * self.w5 * sigmoid_derivative(h1) * X[1]
        dC_dw4 = -2 * (y - o1) * sigmoid_derivative(o1) * self.w6 * sigmoid_derivative(h2) * X[1]

        # Gradient Descent
        self.w5 = self.w5 - learning_rate * dC_dw5
        self.w6 = self.w6 - learning_rate * dC_dw6
        self.w1 = self.w1 - learning_rate * dC_dw1
        self.w2 = self.w2 - learning_rate * dC_dw2
        self.w3 = self.w3 - learning_rate * dC_dw3
        self.w4 = self.w4 - learning_rate * dC_dw4

    def train(self, X, y , epochs=10000, learning_rate=0.1):
        for epoch in range(epochs):
            h1, h2, o1 = self.feedforward(X)
            if epoch == 0:
                o1_initial = o1
            self.backpropagation(X, y, h1, h2, o1, learning_rate)
            loss = mse_loss(y, o1)
            print(f"Epoch: {epoch}, Loss: {loss}")

        return o1, o1_initial
    
if __name__ == "__main__":
    my_own_neural_network = NeuralNetwork()
    X = np.array([0.5, 0.3])    # input of x1 and x2
    y = np.array([1])           # expected output 1 = dog ; 0 = not dog
    o1, o1_initial = my_own_neural_network.train(X, y)

    print(f"Model's Output Before Training: {o1_initial}")
    print(f"Model's Output After Training: {o1[0]}")
    print(f"Desired Output From Model: {y[0]}")