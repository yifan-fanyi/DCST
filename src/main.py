# 2020.01.27
# @yifan
# main function
#
import numpy as np
import cv2
import copy
from skimage.measure import block_reduce
import warnings
warnings.filterwarnings("ignore")

from color_space import BGR2YUV, ML_inv_color, YUV2BGR, BGR2PQR, PQR2BGR
from utli import Clip, write_to_txt
from transform import myPCA, Shrink, invShrink
from quantize import compute_quantization_matrix, Q, dQ
from evaluate import PSNR

def pipeline(X, Qstep=16, ML_inv=True, write=False, name='tmp.txt', tPCA=None):
    H, W = X.shape[0], X.shape[1]
    X = X.reshape(1, H, W, -1)
    X_p, X_q, X_r = X[:,:,:,0:1], block_reduce(X[:,:,:,1:2], (1, 2, 2, 1), np.mean), block_reduce(X[:,:,:,2:], (1, 2, 2, 1), np.mean)
    # P
    def proP(X_p, Qstep):
        X_block = Shrink(X_p, win=8)
        trans_pca = myPCA(is2D=True, H=8, W=8)
        trans_pca.fit(X_block)
        tX = trans_pca.transform(X_block)
        tX = Q(trans_pca, tX, Qstep)
        return trans_pca, tX
        # Quant
    def inv_proP(trans_pca, tX, Qstep, Xraw, ML_inv):
        Xraw = Shrink(Xraw, win=8)
        tX = dQ(trans_pca, tX, Qstep)
        if ML_inv == True:
            iX = trans_pca.ML_inverse_transform(Xraw, tX)
        else:
            iX = trans_pca.inverse_transform(tX)
        iX_p = invShrink(iX, win=8)
        return iX_p

    # QR
    def proQR(trans_pca, X_q, Qstep):
        X_block = Shrink(X_q, win=8)
        tX = trans_pca.transform(X_block)
        tX = Q(trans_pca, tX, Qstep)
        return tX
        
    if tPCA == None:
        tPCA, tX_p = proP(X_p, Qstep)
    else:
        tX_p = proQR(tPCA, X_p, Qstep)
    if write == True:
        write_to_txt(tX_p, name)
    iX_p = inv_proP(tPCA, tX_p, Qstep, X_p, ML_inv)

    tX_q = proQR(tPCA, X_q, Qstep)
    if write == True:
        write_to_txt(tX_q, name)
    iX_q = inv_proP(tPCA, tX_q, Qstep, X_q, ML_inv)
    iX_q = cv2.resize(iX_q[0,:,:,0], (W, H)).reshape(1, H, W, 1)

    tX_r = proQR(tPCA, X_r, Qstep)
    if write == True:
        write_to_txt(tX_r, name)
        with open(name, 'a') as f:
            f.write('-1111')
    iX_r = inv_proP(tPCA, tX_r, Qstep, X_r, ML_inv)
    iX_r = cv2.resize(iX_r[0,:,:,0], (W, H)).reshape(1, H, W, 1)
    
    return np.concatenate((iX_p, iX_q, iX_r), axis=-1)
    

def run(tPCA=None, ML_inv=True, ML_color_inv=True, img=0, write=False, YUV=True, name='None'):
    psnr = []
    q = [200, 160, 140, 120, 100, 90, 80, 70, 60, 50, 40.0,32.0,26.6,22.8,20.0,17.7,16.0,14.4,12.8,11.2,9.6,8.0,6.4,4.8,3.2]
    for i in range(len(q)):     
        X_bgr = cv2.imread('/Users/alex/Desktop/proj/compression/data/Kodak/Kodak/'+str(img)+'.bmp')
        if YUV == False:
            color_pca, X = BGR2PQR(X_bgr)
        else:
            X = BGR2YUV(X_bgr)
        iX = pipeline(X, Qstep=q[i], ML_inv=ML_inv, write=write, name='../result/'+name+'/'+str(img)+'_'+str(i)+'.txt', tPCA=tPCA)
        if ML_color_inv == True:
            iX = ML_inv_color(X_bgr, iX)
            #cv2.imwrite(str(img)+'_'+str(i)+'.png', copy.deepcopy(iX))
            #cv2.imwrite(str(img)+'_'+str(i)+'.png', BGR2RGB(copy.deepcopy(iX)))
            psnr.append(PSNR(iX, X_bgr))
        else:
            if YUV == True:
                iX = YUV2BGR(iX)
            else:
                iX = PQR2BGR(iX), color_pca
            psnr.append(PSNR(iX[0], X_bgr))
        #break
    return psnr

if __name__ == "__main__":
    psnr = []
    name = 'our_pqr_tmp'
    for i in range(24):
        psnr.append(run(None, 1, 1, img=i, write=0, YUV=False, name=name))
    psnr = np.array(psnr)
    with open('../result/psnr_'+name+'.pkl', 'wb') as f:
        pickle.dump(psnr,f)