import numpy as np
from RVFL import RVFL_train
import scipy.io

file_data = scipy.io.loadmat('./aus.mat')

file_data1 = file_data['X1']
file_data2 = file_data['X2']
label = file_data['y']

m, n = file_data1.shape
for i in range(m):
    if label[i] == -1:
        label[i] = 0

file_data1= np.hstack((file_data1, label))
file_data2= np.hstack((file_data2, label))
np.random.seed(0)
indices = np.random.permutation(m)
#View A
file_data1 = file_data1[indices]
A_train=file_data1[0:int(m*(1-0.30))]
A_test=file_data1[int(m * (1-0.30)):]

#View 2
file_data2 = file_data2[indices]
B_train=file_data2[0:int(m*(1-0.30))]
B_test=file_data2[int(m * (1-0.30)):] 

C1 = 1
C2 = 1
rho1 = 1
NN = 3

Eval, Test_time = RVFL_train(A_train, B_train, A_test, B_test, C1,C1, C2,rho1, NN)
Test_accuracy=Eval[0]
print('Test Accuracy:', Test_accuracy)