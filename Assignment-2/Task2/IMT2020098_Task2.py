STUDENT_ROLLNO = "IMT2020098" #yourRollNumberHere
STUDENT_NAME = "Jainav Sangvhi" #yourNameHere
#@PROTECTED_1
##DO NOT MODIFY THE BELOW CODE. NO OTHER IMPORTS ALLOWED. NO OTHER FILE LOADING OR SAVING ALLOWED.
from torch import Tensor
import torch.nn as nn 
import torch.optim as optim 
import torchmetrics
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection 
import torch.utils.data as data 
from torch.utils.data import TensorDataset, DataLoader
import numpy as np 
import torch
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
submission = np.load("sample_submission.npy")
#@PROTECTED_1
y_test = np.load("y_test.npy")

def OneHEncoding(y):
    temp = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        temp[i][int(y[i])] = 1 
    temp = torch.from_numpy(temp.astype(np.float32))
    return temp

def Normalization(X):
  X = X/255
  return X


def calc_cost(y_act, y_pred):
    e = y_act - y_pred
    loss = np.mean(e**2)
    return loss

def train(X, y):
  Bsize=64
  costFunction = nn.MSELoss()
  opt = optim.SGD(model.parameters(), lr=0.01)
  costHistory = []
  for i in range(5100):
    s = i*Bsize
    e = (i+1)*Bsize
    Xn = X[s:e]
    yn = y[s:e]
    for (x, y) in zip(Xn, yn):
      opt.zero_grad()
      y_pred = model(x)
      loss = costFunction(y_pred, y)
      costHistory.append(loss.item())
      loss.backward()
      opt.step()

def test(X,y):
  with torch.no_grad():
    res = model(X)
  submission = res.argmax(axis=1).numpy()
  return 100- calc_cost(y, submission)

input_nuerons = X_train.shape[1]
hidden_neurons = 256
output_neurons = 10

X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))

model = nn.Sequential(nn.Linear(input_nuerons, hidden_neurons), nn.Tanh(), nn.Linear(hidden_neurons, output_neurons), nn.Tanh())

X_train = Normalization(X_train)
y_train = OneHEncoding(y_train)
X_test = Normalization(X_test)

train(X_train, y_train)
print("Total Accuracy= ", test(X_test,y_test))
#@PROTECTED_2
np.save("{}__{}".format(STUDENT_ROLLNO,STUDENT_NAME),submission)
#@PROTECTED_2