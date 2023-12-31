#implement gradient descent on MNIST dataset

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_normalized_data():
    print("Reading in and transforming data...")

    if not os.path.exists(r'C:\Users\zygog\OneDrive\Desktop\Data_Science_Modern_Deep_Learning_in_Python\kaggle_MNIST\train.csv'):
        print(r'Looking for C:\Users\zygog\OneDrive\Desktop\Data_Science_Modern_Deep_Learning_in_Python\kaggle_MNIST\train.csv')
        print('You have not downloaded the data and/or not placed the files in the correct location.')
        print('Please get the data from: https://www.kaggle.com/c/digit-recognizer')
        print('Place train.csv in the folder large_files adjacent to the class folder')
        exit()

    df = pd.read_csv(r'C:\Users\zygog\OneDrive\Desktop\Data_Science_Modern_Deep_Learning_in_Python\kaggle_MNIST\train.csv')
    data = df.values.astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 1:]
    Y = data[:, 0]

    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:]
    Ytest = Y[-1000:]

    # normalize the data
    mu = Xtrain.mean(axis=0)
    std = Xtrain.std(axis=0)
    np.place(std, std == 0, 1)
    Xtrain = (Xtrain - mu) / std
    Xtest = (Xtest - mu) / std

    return Xtrain, Xtest, Ytrain, Ytest



def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    K= y.max() +1
    ind = np.zeros((N, K))
    for i in range(N):
        k = y[i]
        ind[i, k] = 1
    return ind

def forward(X, W, b):
    # softmax
    a = X.dot(W) + b
    expa = np.exp(a)
    y = expa / expa.sum(axis=1, keepdims=True)
    return y

def cost(p_y, t):
    tot = t * np.log(p_y)
    return -tot.sum()

def predict(p_y):
    return np.argmax(p_y, axis=1)

def error_rate(p_y, t):
    prediction = predict(p_y)
    return np.mean(prediction != t)

def gradW(t, y, X):
    return X.T.dot(t - y)

def gradb(t, y):
    return (t - y).sum(axis=0)



def linear_benchmark():
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    print("Performing logistic regression...")

    N,D = Xtrain.shape
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)
    K = Ytrain_ind.shape[1]

    W = np.random.randn(D,K) / np.sqrt(D)
    b = np.zeros(K)
    train_losses = []
    test_losses = []
    train_classification_errors = []
    test_classification_errors = []

    lr = 0.00003
    reg = 0.0
    n_iters = 100
    for i in range(n_iters):
        p_y = forward(Xtrain, W, b)
        train_loss = cost(p_y, Ytrain_ind)
        train_losses.append(train_loss)

        train_err = error_rate(p_y, Ytrain)
        train_classification_errors.append(train_err)

        p_y_test = forward(Xtest, W, b)
        test_loss = cost(p_y_test, Ytest_ind)
        test_losses.append(test_loss)

        test_err = error_rate(p_y_test, Ytest)
        test_classification_errors.append(test_err)

        W += lr * (gradW(Ytrain_ind, p_y, Xtrain) - reg * W)
        b += lr * gradb(Ytrain_ind, p_y)
        if (i+1) % 10 == 0:
           print(f"Iter: {i+1}/{n_iters}, Train loss: {train_loss:.3f} "
                 f"Train error: {train_err:.3f}, Test loss: {train_loss:.3f} "
                 f"Test error: {test_err:.3f}")

    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))

    plt.plot(train_losses, label = 'Train loss')
    plt.plot(test_losses, label = 'Test loss')
    plt.title("Loss per iteration")
    plt.legend()
    plt.show()

    plt.plot(train_classification_errors, label='Train error')
    plt.plot(test_classification_errors, label='Test error')
    plt.title("Classification error per iteration")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    linear_benchmark()
