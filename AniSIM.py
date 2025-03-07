import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Basic functions
def sqrtm_svd(A):
    # Singular value decomposition
    U, s, Vt = np.linalg.svd(A)
    
    # Construct the diagonal matrix of the square roots of the singular values
    sqrt_s = np.diag(np.sqrt(s))
    
    # Compute the square root of the matrix
    sqrt_A = U @ sqrt_s @ Vt
    
    return sqrt_A

def generate_spiked_covariance_matrix(beta, lambda_val):
    beta = beta / np.linalg.norm(beta)  # Normalize beta
    identity_matrix = np.eye(len(beta))
    spike_matrix = lambda_val * np.outer(beta, beta)
    covariance_matrix = (identity_matrix + spike_matrix)/(1+lambda_val)
    return covariance_matrix

def H3s(X, beta):
    single_index = X @ beta
    y = single_index**3-3*single_index
    return y

def H2s(X, beta):
    single_index = X @ beta
    y = single_index**2-1
    return y

def Sign(X, beta):
    single_index = X @ beta
    y = np.sign(single_index)
    return y

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

# Main algorithms

# Vanilla SGD 

def vanilla_SGD(X, y, cov_matrix, beta, learning_rate=0.0001):
    n, d = X.shape
    r = 0.1 # Initialization scaling
    w = r*np.random.randn(d)/d # hidden layer weight
    sQ = sqrtm_svd(cov_matrix) # square root of the covariance matrix
    if np.dot(w/ np.linalg.norm(sQ@w), cov_matrix@beta)<0:
      w=-w
          
    correlations = [] # correlation measured in the Euclidean geometry
    weight_norms = [] # norm of w
    wsig = [] # signal component of w
    corQ =[] # correlation measured in the geometry induced by Q
    wpe = [] # noise component of w
    
    for i in range(n):
            xi = X[i]
            yi = y[i]
      
            # Compute the gradients
            zi = np.dot(w, xi)
            grad_w = -yi *  relu_derivative(zi) * xi

            # Clip gradients to avoid overflow
            grad_w = np.clip(grad_w, -1, 1)
          
            # Update the parameters
            w -= learning_rate * grad_w
        
            # Compute and store the correlation between w and beta
            correlations.append(np.dot(w / np.linalg.norm(w), beta/ np.linalg.norm(beta)))

            # Compute and store the Q_correlation
            corQ.append(np.dot(w/ np.linalg.norm(sQ@w), cov_matrix@beta))

            # Store the norm of the weight vector
            weight_norms.append(np.linalg.norm(w))
        
            # Compute and store the scalar product between w and the model direction
            wsig.append(np.dot(w, beta))

            # Compute and store wpe (noise component of the weight)
            wpe.append(np.sqrt(np.linalg.norm(w)**2-np.dot(w, beta)**2))

            # Update the learning rate
            #learning_rate=max(learning_rate* (1+0.000001), 0.0001)
    
    return w, correlations, weight_norms, wsig, corQ, wpe

# Spherical SGD

def Spherical_SGD(X, y, cov_matrix, beta, learning_rate=0.0001):
    n, d = X.shape
    w = np.random.randn(d)
    sQ = sqrtm_svd(cov_matrix)
    w = w/np.linalg.norm(sQ@w)
    if np.dot(w, cov_matrix@beta)<0:
      w=-w
          
    correlations = []
    weight_norms = []
    wsig = []
    corQ =[]
    wpe = []
    

    for i in range(n):
        xi = X[i]
        yi = y[i]
        zi = np.dot(w, xi)
            
        # Compute the gradients
        grad_w = -yi * relu_derivative(zi) * xi
        sgrad_w = grad_w - (np.dot(grad_w, sQ @ w) / np.dot(sQ @ w, sQ @ w)) * (sQ @ w)

        # Clip gradients to avoid overflow
        sgrad_w = np.clip(sgrad_w, -1, 1)
          
        # Update the parameters
        w -= learning_rate * sgrad_w
        w = w/np.linalg.norm(sQ@w)
      
        # Compute and store the correlation between w and beta
        correlations.append(np.dot(w / np.linalg.norm(w), beta/ np.linalg.norm(beta)))

        # Compute and store the Q_correlation
        corQ.append(np.dot(w/ np.linalg.norm(sQ@w), cov_matrix@beta))

        # Store the norm of the weight vector
        weight_norms.append(np.linalg.norm(w))
        
        # Compute and store the scalar product between w and the model direction
        wsig.append(np.dot(w, beta))

        # Compute and store wpe (noise component of the weight)
        wpe.append(np.sqrt(np.linalg.norm(w)**2-np.dot(w, beta)**2))
    
    return w, correlations, weight_norms,wsig, corQ, wpe

# Batch Re-use

def RepSGD(X, y, cov_matrix, beta, lr1=0.001, lr2= 0.001):
    n, d = X.shape
    r = 0.1
    w = r*np.random.randn(d)/d
    sQ = sqrtm_svd(cov_matrix)
    if np.dot(w/ np.linalg.norm(sQ@w), cov_matrix@beta)<0:
      w=-w
      
    correlations = []
    weight_norms = []
    wsig = []
    corQ =[]
    wpe = []
    
    for i in range(n):
        xi = X[i]
        yi = y[i]
        zi = np.dot(w, xi)
      
        # First Gradient Step
        grad_w = -yi * relu_derivative(zi) * xi

        # Clip gradients to avoid overflow
        grad_w = np.clip(grad_w, -1, 1)
      
        # Update the parameters
        w -= lr1 * grad_w

        # Compute the model output2
        zi = np.dot(w, xi)

        # Second Gradient Step
        grad_w = -yi * relu_derivative(zi) * xi
        grad_w = np.clip(grad_w, -1, 1)
      
        # Update the parameters
        w -= lr2 * grad_w
                
        # Compute and store the correlation between w and beta
        correlations.append(np.dot(w / np.linalg.norm(w), beta/ np.linalg.norm(beta)))

        # Compute and store the Q_correlation
        corQ.append(np.dot(w/ np.linalg.norm(sQ@w), cov_matrix@beta))

        # Store the norm of the weight vector
        weight_norms.append(np.linalg.norm(w))
    
        # Compute and store the scalar product between w and the model direction
        wsig.append(np.dot(w, beta))

        # Compute and store the scalar product between w and the model direction
        wpe.append(np.sqrt(np.linalg.norm(w)**2-np.dot(w, beta)**2))

    return w, correlations, weight_norms,wsig, corQ, wpe

