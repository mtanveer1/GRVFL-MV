import numpy as np
import time
from Graph import calc_ge_data2

def relu(x):
    return np.maximum(0, x)

def Evaluate(ACTUAL, PREDICTED):
    idx = (ACTUAL == 1)
    p = np.sum(idx)
    n = np.sum(~idx)
    N = p + n
    tp = np.sum(np.logical_and(ACTUAL[idx] == 1, PREDICTED[idx] == 1))
    tn = np.sum(np.logical_and(ACTUAL[~idx] == 0, PREDICTED[~idx] == 0))
    accuracy = 100 * (tp + tn) / N
    
    EVAL = [accuracy]
    return EVAL

def one_hot(x, n_class):
    y = np.zeros([len(x), n_class])
    U_dataY_train = np.array([0,1])
    for i in range(n_class):
        idx= (x == U_dataY_train[i])
        y[idx,i]=1
    return y


def RVFL_train(train_data1,train_data2, test_data1,test_data2, theta2,c2,c3, rho,N):
    # np.random.seed(2)
    theta1=theta2
    c1=c2
    start = time.time()
    trainX=train_data1[:,:-1]
    trainY=train_data1[:,-1]

    trainX2=train_data2[:,:-1]

    s = 0.1
    
    Nsample, Nfea = trainX.shape
    Nsample2, Nfea2 = trainX2.shape
    nclass = 2
    dataY_train_temp = one_hot(trainY,nclass)

    trainY[trainY==0]=2
    

    #View 1
    W = np.random.rand(Nfea, N) * 2 - 1
    b = s * np.random.rand(1, N)
    X1 = np.dot(trainX, W) + np.tile(b, (Nsample, 1))
    X1=relu(X1)
    X = np.concatenate((trainX, X1), axis=1)
    X_A = np.hstack((X, np.ones((Nsample, 1))))  # Bias in the output layer

    S1=calc_ge_data2(X_A.T, trainY)
    #View 2
        
    Nsample2, Nfea2 = trainX2.shape  
    W2 = np.random.rand(Nfea2, N) * 2 - 1
    b2 = s * np.random.rand(1, N)
    X1 = np.dot(trainX2, W2) + np.tile(b2, (Nsample2, 1))
    X1=relu(X1)
    X = np.concatenate((trainX2, X1), axis=1)
    X_B = np.hstack((X, np.ones((Nsample2, 1))))  # Bias in the output layer
    S2=calc_ge_data2(X_B.T, trainY)


    X_mat_A = np.concatenate((c3*np.eye(X_A.shape[1])+theta1*S1+c1*np.dot(X_A.T, X_A), rho*np.dot(X_A.T, X_B)), axis=1)
    X_mat_B = np.concatenate((rho*np.dot(X_B.T, X_A), np.eye(X_B.shape[1])+theta2*S2+c2*np.dot(X_B.T, X_B)), axis=1)
    X_mat = np.vstack((X_mat_A, X_mat_B))

    X_mat_Rhs= np.dot(np.vstack((X_A.T*(c1+rho), X_B.T*(c2+rho))), dataY_train_temp)

    beta = np.dot(np.linalg.inv(X_mat), X_mat_Rhs)
    hl=Nfea+N+1
    beta1 = beta[:hl]
    beta2 = beta[hl:]

    end = time.time()
    # Test_A

    T_A=test_data1[:,:-1]
    Y=test_data1[:,-1]
    
    Nsample = T_A.shape[0]

    # Test Data

    X1 = np.dot(T_A, W) + np.tile(b, (Nsample, 1))
    X1=relu(X1)

    X1 = np.hstack((X1, np.ones((Nsample, 1))))
    XZ1 = np.hstack((T_A, X1))
    rawScore1 = np.dot(XZ1, beta1)
    Validation_label1 = np.argmax(rawScore1, axis=1) 

    #Test_B
    T_B=test_data2[:,:-1]
    
    Nsample = T_B.shape[0]

    # Test Data

    X1 = np.dot(T_B, W2) + np.tile(b2, (Nsample, 1))
    X1=relu(X1)

    X1 = np.hstack((X1, np.ones((Nsample, 1))))
    XZ2 = np.hstack((T_B, X1))

    rawScore2 = np.dot(XZ2, beta2)
    Validation_label2 = np.argmax(rawScore2, axis=1) 

    rawScore3 = np.dot(XZ2, beta2) + np.dot(XZ1, beta1)
    Validation_label3 = np.argmax(rawScore3, axis=1) 

    Y=Y.reshape(Y.shape[0], 1)
    Validation_label1=Validation_label1.reshape(Validation_label1.shape[0], 1)
    Validation_label2=Validation_label2.reshape(Validation_label1.shape[0], 1)
    Validation_label3=Validation_label3.reshape(Validation_label1.shape[0], 1)

    EVAL_Validation1 = Evaluate(Y, Validation_label1)
    EVAL_Validation2 = Evaluate(Y, Validation_label2)
    EVAL_Validation3 = Evaluate(Y, Validation_label3)
   
    # EVAL_Validation_majorityvoting = Evaluate(Y, indx)
    Evala=np.vstack((EVAL_Validation1,EVAL_Validation2))
    Eval= np.vstack((Evala, EVAL_Validation3))
    idx = np.array((EVAL_Validation1[0],EVAL_Validation2[0],EVAL_Validation3[0]))
    ad=idx.argmax()
    EVAL_Validation=Eval[ad]
    
    Time=end - start
    return EVAL_Validation,Time
