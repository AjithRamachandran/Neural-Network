import numpy as np

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

def propogate(w1, w2, b1, b2, X, y):
    m = X.shape[1]
    
    #Forward Propagation
    z1 = np.dot(w1, X)
    a1 = sigmoid(z1 + b1)
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
        
    params = {"w1": w1,
              "b1": b1,
              "w2": w2,
              "b2": b2}
    
    return params

def predict(X, params):
    a1 = np.dot(params['w1'], X) + params['b1']
    a2 = np.dot(params['w2'], a1) + params['b2']
    
    a2 = np.squeeze(a2)
    
    if(a2>0.5):
        return 1
    else:
        return 0


def main():
    X = np.array([[1,2,1,1,0,1], [2,3,3,7,2,3], [3,4,5,3,4,9]])
    y = np.array([[0,0,1,1,0,1]])

    w1, w2, b1, b2 = initialize(X.shape[0], 8, y.shape[0])
    params = optimize(w1, w2, b1, b2, X ,y, epoch=5, learning_rate=0.4)
    print(predict(np.array([[1],[4],[5]]), params))

if __name__ == '__main__':
    main()