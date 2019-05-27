import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def initialize(n_x, n_h, n_y):
    np.random.seed(3)
    
    w1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))
    
    return w1, w2, b1, b2

def predict(X, params, n_val):
    a1 = np.dot(params['w1'], X) + params['b1']
    a2 = np.dot(params['w2'], a1) + params['b2']

    for i in range(n_val):
        if(a2[0][i]>0.5):
            a2[0][i] = 1
        else:
            a2[0][i] = 0

    return a2

def score(yval, result, n_val):
    count = 0

    for i in range(n_val):
        if(yval[0][i] == result[0][i]):
            count += 1
    
    score = (count/n_val)*100

    return score

def propogate(w1, w2, b1, b2, X, y):
    m = X.shape[1]
    
    #Forward Propagation
    z1 = np.dot(w1, X)
    a1 = tanh(z1 + b1)
    z2 = np.dot(w2, a1)
    a2 = sigmoid(z2 + b2)
    cost = -np.sum(np.multiply(np.log(a2), y) + np.multiply((1 - y), np.log(1 - a2)))/m
    
    #Backward Propagation
    dz2 = a2-y
    dw2 = np.dot(dz2, a1.T)/m
    db2 = np.sum(dz2, axis=1, keepdims=True)/m
    dz1 = np.multiply(np.dot(w2.T, dz2), 1-np.power(a1, 2))
    dw1 = np.dot(dz1, X.T)/m
    db1 = np.sum(dz1, axis=1, keepdims=True)/m
    
    assert(dw1.shape == w1.shape)
    assert(dw2.shape == w2.shape)
    
    cost = np.squeeze(cost)
    
    grads = {"dw1": dw1,
             "db1": db1,
             "dw2": dw2,
             "db2": db2}
             
    return grads, cost

def optimize(w1, w2, b1, b2, X ,y, epoch=1000, learning_rate=0.001):
    loss = []
    for i in range(epoch):
        grads, cost = propogate(w1, w2, b1, b2, X, y)
        dw1 = grads["dw1"]
        db1 = grads["db1"]
        dw2 = grads["dw2"]
        db2 = grads["db2"]
        
        w1 = w1 - learning_rate*dw1
        b1 = b1 - learning_rate*db1
        w2 = w2 - learning_rate*dw2
        b2 = b2 - learning_rate*db2
        
        print("After ", i+1, "epoch, loss:", cost)
        loss.append(cost)
        
    params = {"w1": w1,
              "b1": b1,
              "w2": w2,
              "b2": b2}
    
    plt.plot(np.squeeze(loss))
    plt.ylabel('Loss')
    plt.xlabel('Iter')
    plt.title("Lr =" + str(learning_rate))
    plt.show()

    return params

def data_prep():
    df = pd.read_csv('wisconsin-cancer-dataset.csv',header=None)
    df = df[~df[6].isin(['?'])]
    df.iloc[:,10].replace(2, 0,inplace=True)
    df.iloc[:,10].replace(4, 1,inplace=True)

    df.head(3)
    scaled_df=df
    names = df.columns[0:10]
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(df.iloc[:,0:10]) 
    scaled_df = pd.DataFrame(scaled_df, columns=names)

    X=scaled_df.iloc[0:500,1:10].values.transpose()
    y=df.iloc[0:500,10:].values.transpose()

    Xval=scaled_df.iloc[501:683,1:10].values.transpose()
    yval=df.iloc[501:683,10:].values.transpose()

    n_val = 182

    return X, y, Xval, yval, n_val

def main():
    X, y, Xval, yval, n_val = data_prep()

    w1, w2, b1, b2 = initialize(X.shape[0], 12, y.shape[0])
    params = optimize(w1, w2, b1, b2, X ,y, epoch=100000, learning_rate=0.003)
    result = predict(Xval, params, n_val)

    score_ = score(yval, result, n_val)

    print(score_)

if __name__ == '__main__':
    main()