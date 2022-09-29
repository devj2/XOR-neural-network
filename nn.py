import numpy as np
# np.random.seed(2)


def tanh(Z):
    return np.tanh(Z)


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def derivSigmoid(A):
    return 1 - np.power(A, 2)


class NN:

    def __init__(self, n1, n2, n3, epochs, alpha, g1, g2, derivg1) -> None:
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.epochs = epochs
        self.learning_rate = alpha
        self.initalise_parameters(n1, n2, n3)
        self.g1 = g1
        self.g2 = g2
        self.derivg1 = derivg1
        
        
    def initalise_parameters(self, n1, n2, n3):
        self.W1 = np.random.rand(n2, n1) - 0.5
        self.W2 = np.random.rand(n3, n2) - 0.5
        self.b1 = np.random.rand(n2, 1) - 0.5
        self.b2 = np.random.rand(n3, 1) - 0.5


    def forward_propagation(self, X):
        S1 = self.W1.dot(X.T) + self.b1
        A1 = self.g1(S1)
        S2 = self.W2.dot(A1) + self.b2
        A2 = self.g2(S2)

        return S1, A1, S2, A2


    def backward_propagation(self, x, y, A1, A2, num_examples):
        y = y.T
        
        errorL2 = A2 - y
        derivW2 = np.dot(errorL2, A1.T) / num_examples
        derivb2 = np.sum(errorL2) / num_examples

        errorL1 = np.multiply(np.dot(self.W2.T, errorL2), self.derivg1(A1))
        derivW1 = np.dot(errorL1, x) / num_examples
        derivb1 = np.sum(errorL1) / num_examples

        return derivW1, derivb1, derivW2, derivb2


    def update_parameters(self, derivW1, derivb1, derivW2, derivb2):
        self.W1 = self.W1 - self.learning_rate * derivW1
        self.b1 = self.b1 - self.learning_rate * derivb1
        self.W2 = self.W2 - self.learning_rate * derivW2
        self.b2 = self.b2 - self.learning_rate * derivb2
    

    def train(self, X, Y):
        print(f'Training on {len(X)} examples')
        print(f'Training on {self.epochs} epochs')
        print('Training started')
        num_examples = len(X)
        
        for epoch in range(1, self.epochs+1):
            print(f'Epoch: {epoch}')
            S1, A1, S2, A2 = self.forward_propagation(X)
            derivW1, derivb1, derivW2, derivB2 = self.backward_propagation(X, Y, A1, A2, num_examples)
            self.update_parameters(derivW1, derivb1, derivW2, derivB2)
            
        print('Training finished')

    def predict(self, x):
        S1 = np.dot(self.W1, x.T) + self.b1
        A1 = self.g1(S1)

        S2 = np.dot(self.W2, A1) + self.b2
        A2 = self.g2(S2)

        prediction = np.squeeze(A2)

        if (prediction >= 0.5):
            return 1
        else:
            return 0



def generate_dataset(num_examples):
    X = np.zeros((1000, 2))
    Y = np.zeros((1000, 1))

    for i in range(1000):
        x0 = np.random.choice([0,1])
        x1 = np.random.choice([0,1])
        X[i] = np.array([x0, x1])
        
        y = x0 ^ x1
        Y[i] = np.array([y])

    return X, Y



def validate_model(X, Y, model):
    correct = 0

    for i in range(1000):
        x0 = np.random.choice([0,1])
        x1 = np.random.choice([0,1])
        x = np.array([x0, x1])
        x.shape = (1, 2)
        yhat = model.predict(x)
    
        y = x0 ^ x1
        if (yhat == y):
            correct += 1

    accuracy = correct / len(Y)
    
    return correct, accuracy


def __main__():
    X_train, Y_train = generate_dataset(1000)
    X_validation, Y_validation = generate_dataset(1000)

    model = NN(2, 2, 1, 10000, 0.15, tanh, sigmoid, derivSigmoid)
    model.train(X_train, Y_train)


    correct, accuracy = validate_model(X_validation, Y_validation, model)
    print(f'Correct: {correct}')
    print(f'Accuracy: {correct/1000}')


if __name__ == '__main__':
    __main__()