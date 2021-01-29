# 2020.01.28
# @yifan
# color space conversion
#
import numpy as np
import copy
from sklearn.preprocessing import normalize

from transform import myPCA
from utli import Clip

def BGR2PQR(X):
    def reScale(K):
        K[:1] = normalize(K[:1], norm='l1')
        K[0,0] *= 219/255
        K[0,1] *= 219/255
        K[0,2] *= 219/255
        sb = 224/255/(np.sum(np.abs(K[1])))
        K[1] *= sb
        sc = 224/255/(np.sum(np.abs(K[2])))
        K[2] *= sc
        return K
    pca = myPCA()
    S = X.shape
    X = X.reshape(-1, 3)
    pca.fit(X)
    pca.Kernels = reScale(pca.Kernels)
    return pca, pca.transform(X).reshape(S)

def PQR2BGR(X, pca):
    return pca.inverse_transform(X, K=np.linalg.inv(pca.Kernels))

def BGR2RGB(X):
    R        = copy.deepcopy(X[:,:,2])
    X[:,:,2] = X[:,:,0]
    X[:,:,0] = R
    return X

def BGR2YUV(X):
    X = BGR2RGB(X)
    K = np.array([[   0.299,    0.587,    0.114],
                  [-0.14713, -0.28886,    0.436],
                  [   0.615, -0.51499, -0.10001]])
    X = np.moveaxis(X, -1, 0)
    S = X.shape
    X = np.dot(K, X.reshape(3, -1))
    X = X.reshape(S)
    X = np.moveaxis(X, 0, -1)
    return X

def YUV2BGR(X):
    K = np.array([[1,        0,  1.13983],
                  [1, -0.39465, -0.58060],
                  [1,  2.03211,        0]])
    X = np.moveaxis(X, -1, 0)
    S = X.shape
    X = np.dot(K, X.reshape(3, -1))
    X = np.moveaxis(X.reshape(S), 0, -1)
    return BGR2RGB(X)

def ML_inv_color(X_bgr, iX):
    llsr = LLSR(onehot=False)
    llsr.fit(iX.reshape(-1,3), X_bgr.reshape(-1,3))
    iX = llsr.predict_proba(iX.reshape(-1,3)).reshape(X_bgr.shape)
    iX = Clip(iX.astype('int32'))
    return iX

class LLSR():
    def __init__(self, onehot=True, normalize=False):
        self.onehot = onehot
        self.normalize = normalize
        self.weight = []

    def fit(self, X, Y):
        if self.onehot == True:
            Y = np.eye(len(np.unique(Y)))[Y.reshape(-1)]
        A = np.ones((X.shape[0], 1))
        X = np.concatenate((X, A), axis=1)
        self.weight, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        return self

    def predict(self, X):
        pred = self.predict_proba(X)
        return np.argmax(pred, axis=1)

    def predict_proba(self, X):
        A = np.ones((X.shape[0], 1))
        X = np.concatenate((X, A), axis=1)
        pred = np.matmul(X, self.weight)
        if self.normalize == True:
            pred = (pred - np.min(pred, axis=1, keepdims=True))/ np.sum((pred - np.min(pred, axis=1, keepdims=True) + 1e-15), axis=1, keepdims=True)
        return pred

    def score(self, X, Y):
        pred = self.predict(X)
        return accuracy_score(Y, pred)