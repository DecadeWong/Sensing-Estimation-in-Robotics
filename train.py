import numpy as np
import cv2
from numpy import linalg as lin

blue_barrel = np.load('blue_barrel.npy')
not_blue = np.load('not_blue.npy')

bb_spl = blue_barrel[np.random.choice(blue_barrel.shape[0],30000,False),:]
#bb_spl = np.array([bb_spl]).T

bb_spl = np.hstack((bb_spl, np.ones([len(bb_spl),1])))


nb_spl = not_blue[np.random.choice(not_blue.shape[0],30000,False),:]
#nb_spl = np.array([nb_spl]).T

nb_spl = np.hstack((nb_spl, np.negative(np.ones([len(nb_spl),1]))))

train_set = np.concatenate((bb_spl,nb_spl), axis = 0)
np.random.shuffle(train_set)
X, y = np.array(train_set[:,(0,1,2)]), np.array(train_set[:,3])
#X = np.array([X]).T
#add intercept
X = np.hstack((X, np.ones([len(X),1])))
#X_T = X[:,np.newaxis]

def sigmoid(z):
    return 1 / ( 1 + np.exp(-z)) 

def iterate(X, y, W_mle, alpha):
    sum = 0
    for i in range(len(X[0])):
        sum = sum + y[i]*X[i,:]*(1-sigmoid(np.dot(y[i]*X[i,:], W_mle)))
    W_mle = W_mle + alpha*sum
    return W_mle

"""i = 0
W_t = np.array([0,0])
W_t1 = iterate(X, y, W_t, 1)

while (np.absolute(lin.norm(W_t - W_t1)>=0.1)):
    W_t = W_t1
    W_t1 = iterate(X,y,W_t,1)
    i=i+1"""

def log_reg(features, labels, steps):
    
    weights = np.zeros(features.shape[1])
    accuracy = 0
    iters =0
    for step in range(steps):

        old_weights = weights
        weights =  iterate(features, labels, weights, 10)
        
        if (step %10 == 0):
            print ("MLE has updated the parameters to", weights)
        if (np.absolute(lin.norm(old_weights - weights)) <1e-10):
            print ("the parameters are converged to", weights)
            break
        iters += 1
    scores = np.dot(features, weights)
    predictions = np.around(2*sigmoid(scores) - 1)
    for i in range(len(predictions)):
        if predictions[i] == y[i]:
            accuracy += 1
    accuracy = accuracy / len(predictions)
    
    return weights, accuracy, iters

weights, accuracy, iters = log_reg(X,y,500)

np.save('parameters1.npy', weights)