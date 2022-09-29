import numpy as np
from nn import NN

def tanh(Z):
    return np.tanh(Z)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def derivSigmoid(A):
    return 1 - np.power(A, 2)


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