# PCS Paper
### Yifan Wang & Zhanxuan Mei
### 2021.01
#### yifanwang0916@outlook.com
***
#### Based on JPEG framework, modified: 
(1) The color space conversion from YCbCr (420) to PQR (420) (PCA based single image color conversion);    
(2) (2D)^2 PCA based block transformaition;    
(3) HSV quantization matrix design;    
(4) Machine learing optimal inverse kernel (including color space inverse kernel and (2D)^2 PCA inverse kernels);    

Save the quantized coefficients in txt file, then, using JPEG's entropy coder to encode them.

