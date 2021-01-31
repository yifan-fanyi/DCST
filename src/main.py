# 2020.01.27
# @yifan
# main function
#
from skimage.measure import block_reduce

from framework.dependency import *
from framework.utli import *

def pipeline(X, Qstep=16, Q_mode=1, ML_inv=True, write=False, name='tmp.txt', tPCA=None, isDCT=True):
    H, W = X.shape[0], X.shape[1]
    X = X.reshape(1, H, W, -1)
    X_p, X_q, X_r = X[:,:,:,0:1], block_reduce(X[:,:,:,1:2], (1, 2, 2, 1), np.mean), block_reduce(X[:,:,:,2:], (1, 2, 2, 1), np.mean)
    # P
    def proP(X_p, Qstep, Q_mode, isDCT):
        X_block = Shrink(X_p, win=8)
        if isDCT == False:
            trans_pca = myPCA(is2D=True, H=8, W=8)
            trans_pca.fit(X_block)
        else:
            trans_pca = DCT(8,8)
        tX = trans_pca.transform(X_block)
        tX = Q(trans_pca, tX, Qstep, mode=Q_mode)
        return trans_pca, tX
        # Quant
    def inv_proP(trans_pca, tX, Qstep, Xraw, ML_inv, Q_mode):
        Xraw = Shrink(Xraw, win=8)
        tX = dQ(trans_pca, tX, Qstep, mode=Q_mode)
        if ML_inv == True:
            iX = trans_pca.ML_inverse_transform(Xraw, tX)
        else:
            iX = trans_pca.inverse_transform(tX)
        iX_p = invShrink(iX, win=8)
        return iX_p

    # QR
    def proQR(trans_pca, X_q, Qstep, Q_mode):
        X_block = Shrink(X_q, win=8)
        tX = trans_pca.transform(X_block)
        tX = Q(trans_pca, tX, Qstep, mode=Q_mode)
        return tX
        
    if tPCA == None:
        tPCA, tX_p = proP(X_p, Qstep, Q_mode, isDCT)
    else:
        tX_p = proQR(tPCA, X_p, Qstep, Q_mode)
    if write == True:
        write_to_txt(tX_p, name)
    iX_p = inv_proP(tPCA, tX_p, Qstep, X_p, ML_inv, Q_mode)

    tX_q = proQR(tPCA, X_q, Qstep, Q_mode)
    if write == True:
        write_to_txt(tX_q, name)
    iX_q = inv_proP(tPCA, tX_q, Qstep, X_q, ML_inv, Q_mode)
    iX_q = cv2.resize(iX_q[0,:,:,0], (W, H)).reshape(1, H, W, 1)

    tX_r = proQR(tPCA, X_r, Qstep, Q_mode)
    if write == True:
        write_to_txt(tX_r, name)
        with open(name, 'a') as f:
            f.write('-1111')
    iX_r = inv_proP(tPCA, tX_r, Qstep, X_r, ML_inv, Q_mode)
    iX_r = cv2.resize(iX_r[0,:,:,0], (W, H)).reshape(1, H, W, 1)
    return np.concatenate((iX_p, iX_q, iX_r), axis=-1)
    

def run(tPCA=None, ML_inv=True, ML_color_inv=True, img=0, write=False, isYUV=True, name=None, Q_mode=0, isDCT=False):
    psnr = []
    if isDCT == True:
        q = np.arange(5, 99, 5)
    else:
        q = [200, 160, 140, 120, 100, 90, 80, 70, 60, 50, 40.0,32.0,26.6,22.8,20.0,17.7,16.0,14.4,12.8,11.2,9.6,8.0,6.4,4.8,3.2]
    for i in range(len(q)):     
        X_bgr = cv2.imread('/Users/alex/Desktop/proj/compression/data/Kodak/Kodak/'+str(img)+'.bmp')
        if isYUV == False:
            color_pca, X = BGR2PQR(X_bgr)
        else:
            X = BGR2YUV(X_bgr)
        iX = pipeline(X, Qstep=q[i], Q_mode=Q_mode, ML_inv=ML_inv, write=write, name='../result/'+name+'/'+str(img)+'_'+str(i)+'.txt', tPCA=tPCA, isDCT=isDCT)
        if ML_color_inv == True:
            iX = ML_inv_color(X_bgr, iX)
            #cv2.imwrite(str(img)+'_'+str(i)+'.png', copy.deepcopy(iX))
            psnr.append(PSNR(iX, X_bgr))
        else:
            if isYUV == True:
                iX = YUV2BGR(iX)
            else:
                iX = PQR2BGR(iX, color_pca)
            psnr.append(PSNR(iX[0], X_bgr))
        #break
    return psnr
  

if __name__ == "__main__":
    psnr = []
    name = 'hvs'
    for i in range(24):
        psnr.append(run(None, 1, 1, img=i, write=1, isYUV=False, name=name, Q_mode=0, isDCT=False))
        #break

