STUDENT_NAME = 'Jainav Sanghvi' #Put your name
STUDENT_ROLLNO = 'IMT2020098' #Put your roll number
CODE_COMPLETE = True
# set the above to True if you were able to complete the code
# and that you feel your model can generate a good result
# otherwise keep it as False
# Don't lie about this. This is so that we don't waste time with
# the autograder and just perform a manual check
# If the flag above is True and your code crashes, that's
# an instant deduction of 2 points on the assignment.
#
#@PROTECTED_1_BEGIN
## No code within "PROTECTED" can be modified.
## We expect this part to be VERBATIM.
## IMPORTS 
## No other library imports other than the below are allowed.
## No, not even Scipy
import numpy as np 
import pandas as pd 
import sklearn.model_selection as model_selection 
import sklearn.preprocessing as preprocessing 
import sklearn.metrics as metrics 
from tqdm import tqdm # You can make lovely progress bars using this

## FILE READING: 
## You are not permitted to read any files other than the ones given below.
X_train = pd.read_csv("train_X.csv",index_col=0).to_numpy()
y_train = pd.read_csv("train_y.csv",index_col=0).to_numpy().reshape(-1,)
X_test = pd.read_csv("test_X.csv",index_col=0).to_numpy()
submissions_df = pd.read_csv("sample_submission.csv",index_col=0)
mnist_test = pd.read_csv("mnist_test.csv")
#@PROTECTED_1_END

y_test = mnist_test['label']
y_test.reset_index(drop=True,inplace=True)
y_test = y_test.to_numpy().reshape(-1,)

mnist_test = mnist_test.drop(['label'], axis=1)
mnist_test.reset_index(drop=True,inplace=True)
mnist_test = mnist_test.to_numpy()

def OneHEncoding(y):
    temp = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        temp[i][int(y[i])] = 1 
    return temp

y_train=OneHEncoding(y_train)
y_test=OneHEncoding(y_test)

def normalization(X):
    X=X/255
    return X

X_train=normalization(X_train)
X_test=normalization(X_test)


def ReLU(x):
    return np.maximum(0, x)

def dReLU(x):
    return x > 0

def softmax(z):
    z = z - np.max(z, axis = 1).reshape(z.shape[0],1)
    return np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(z.shape[0],1)
    
def sigmoid(Z):
    A = 1/(1+np.exp(np.dot(-1, Z)))
    return A
# layer_dims holds the dimensions of each layer
def InitializeParameters(layer_dims):
    np.random.seed(10)
    params = {}
    L = len(layer_dims)
    for l in range(0, L-1): #0 to 2
        params['W'+str(l+1)] = np.random.randn(layer_dims[l], layer_dims[l+1])
        params['b'+str(l+1)] = np.random.randn(layer_dims[l+1],)      
    return params

def shuffleData(X, Y):
    i = [j for j in range(X.shape[0])]
    np.random.seed(10)
    np.random.shuffle(i)
    X = X[i]
    Y = Y[i]
    return X, Y

def forwardPropogation(X,params):
    A = X # input to first layer i.e. training data
    input = []
    output = []
    L = len(params)//2
    for l in range(1, L+1):
        A_prev = A
        # Linear Hypothesis
        Z = np.dot(A_prev,params['W'+str(l)]) + params['b'+str(l)] 
        input.append(Z)
        # Applying activation function on linear hypothesis
        if l==L: A= softmax(Z) 
        else: A= ReLU(Z) 
        output.append(A)   
    return input,output

def backPropogation(X, Y,input,output, params, bSize, learning_rate):
    error = output[2] - Y
    dCost = (error)/bSize
    dW3 = np.dot(dCost.T,output[1]).T
    dW2 = np.dot((np.dot((dCost),params['W3'].T) * dReLU(input[1])).T,output[0]).T
    dW1 = np.dot((np.dot(np.dot((dCost),params['W3'].T)*dReLU(input[1]), params['W2'].T)*dReLU(input[0])).T,X).T   
    db3 = np.sum(dCost,axis = 0)
    db2 = np.sum(np.dot((dCost),params['W3'].T) * dReLU(input[1]),axis = 0)
    db1 = np.sum((np.dot(np.dot((dCost),params['W3'].T)*dReLU(input[1]),params['W2'].T)*dReLU(input[0])),axis = 0)   
    #Now applying SGD 
    params['W3'] -= learning_rate*dW3
    params['W2'] -= learning_rate*dW2
    params['W1'] -= learning_rate*dW1   
    params['b3'] -= learning_rate*db3
    params['b2'] -= learning_rate*db2
    params['b1'] -= learning_rate*db1
    return params

def train(X, y, layer_dims, lr):
    params = InitializeParameters(layer_dims)
    cost_history = []
    accuracy_history = []
    bSize=64
    learning_rate=lr

    Xn = X[:bSize] # batch input 
    yn = y[:bSize] # batch target value
    for i in range(20):
        X, y = shuffleData(X, y)
        cost = 0
        accuracy = 0
        for index in range(X.shape[0]//bSize-1):
            s = index*bSize
            e = (index+1)*bSize
            Xn = X[s:e]
            yn = y[s:e]
            input,output = forwardPropogation(Xn,params)
            params = backPropogation(Xn, yn,input,output,params, bSize, learning_rate)
            error = output[2] - yn
            cost+=np.mean(error**2)
            accuracy+= np.count_nonzero(np.argmax(output[2],axis=1) == np.argmax(yn,axis=1)) / bSize
        cost_history.append(cost/(X.shape[0]//bSize))
        accuracy_history.append(accuracy*100/(X.shape[0]//bSize))
    return params

params=train(X_train,y_train,[784,512,512,10],0.005)

def test(X_test,y_test,mnist_test,params):
    mnist_test = normalization(mnist_test)
    X_test = normalization(X_test)
    
    input,output = forwardPropogation(mnist_test,params)
    accuracy = np.count_nonzero(np.argmax(output[2],axis=1)==np.argmax(y_test,axis=1)) / mnist_test.shape[0]
    print("Total Accuracy = ", 100 * accuracy, "%")
    
    input,output = forwardPropogation(X_test,params)
    out = np.zeros((output[2].shape[0],))
   
    for i in range(out.shape[0]):
        out[i] = np.argmax(output[2][i])
    res_df = pd.DataFrame(out, columns = ['label'])
    return res_df    

submissions_df = test(X_test,y_test,mnist_test,params)

#@PROTECTED_2_BEGIN 
##FILE WRITING:
# You are not permitted to write to any file other than the one given below.
submissions_df.to_csv("{}__{}.csv".format(STUDENT_ROLLNO,STUDENT_NAME))
#@PROTECTED_2_END