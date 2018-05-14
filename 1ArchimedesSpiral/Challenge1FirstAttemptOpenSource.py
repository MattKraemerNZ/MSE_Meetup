
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier

def plot_labeled_2d(A,dot=','):
    labels = np.unique(A[:,A.shape[-1]-1]).astype(int)
    plt.figure(figsize=[5,5])
    for l in labels:
        plt.plot(A[:,0][A[:,2]==l],A[:,1][A[:,2]==l],dot)
    plt.axis('off')
    plt.show()

mlp = MLPClassifier(hidden_layer_sizes=(100,50,20), max_iter=10000, alpha=1e-4,
                    solver='sgd', verbose=1, tol=1e-5, random_state=1,
                    learning_rate_init=.01)

#You'll need to enter your path to the training files here
datafolder = os.path.join(os.path.expanduser('~'),'Documents',"MSE_Meetup","1ArchimedesSpiral")
print("Looking for files in here: ", datafolder)

df = pd.read_csv(os.path.join(datafolder,"training.csv"),index_col=[0])

X,y = df.values[:3000,:2], df.values[:3000,2]
testdata,correct_labels = df.values[3000:,:2], df.values[3000:,2]
mlp.fit(X,y)

result = np.array([testdata.T[0],testdata.T[1],mlp.predict(testdata)]).T

plot_labeled_2d(result,'.')


# In[ ]:


'''
def identity_func(x): 
    return x

def linear_func(x):
    return 1.5 * x + 1 #(1.5),(1)

def binarystep_func(x):
    return (x>=0)*1
    # return np.array(x>=0, dtype = np.int) # same result
 
    # y = x >= 0
    # return y.astype(np.int) # Copy of the array, cast to a specified type.
    # same result

def sgn_func(x): # (sign function)
    return (x>=0)*1 + (x<=0)*-1

def softstep_func(x): # Soft step (= Logistic),(Sigmoid)
    return 1 / (1 + np.exp(-x))

def tanh_func(x): # TanH
    return np.tanh(x)
    # return 2 / (1 + np.exp(-2*x)) - 1 # same

def arctan_func(x): # ArcTan
    return np.arctan(x)

def softsign_func(x): # Softsign
    return x / ( 1+ np.abs(x) )

def relu_func(x): # ReLU(Rectified Linear Unit
    return (x>0)*x
    # return np.maximum(0,x) # same

def leakyrelu_func(x): # Leaky ReLU(Rectified Linear Unit
    return (x>=0)*x + (x<0)*0.01*x # (0.01)
    # return np.maximum(0.01*x,x) # same

def elu_func(x): # ELU(Exponential linear unit)
    return (x>=0)*x + (x<0)*0.01*(np.exp(x)-1)

def trelu_func(x): # Thresholded ReLU
    return (x>1)*x 

def softplus_func(x): # SoftPlus
    return np.log( 1 + np.exp(x) )

def bentidentity_func(x): # Bent identity
    return (np.sqrt(x*x+1)-1)/2+x

def gaussian_func(x): # Gaussian
    return np.exp(-x*x)

def sigmoid(x):
    return 0.5 * (np.tanh(x / 2.) + 1)
'''

