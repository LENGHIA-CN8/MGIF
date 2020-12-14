import math

import cv2
import numpy as np
from PIL import Image;

def local_mean(arr,r):
    (rows,cols) = arr.shape[:2];
    out = np.zeros((rows,cols))
    mask = np.zeros((2*r+1,2*r+1))
    # truyen cols vaf rows 0 vao anh
    zeros_1 = np.zeros((1,arr.shape[1]))
    for i in range(r):
        arr = np.insert(arr,arr.shape[0],zeros_1,axis=0)
        arr = np.insert(arr,0,zeros_1,axis=0)
    zeros_2 = np.zeros((1,arr.shape[0]))
    for i in range(r):
        arr = np.insert(arr,arr.shape[1],zeros_2,axis=1)
        arr = np.insert(arr,0,zeros_2,axis=1)
    for i in range(arr.shape[0]-(2*r)):
        for j in range(arr.shape[1]-(2*r)):
            # print(j)
            mask = arr[i:i+2*r+1,j:j+2*r+1]
            out[i,j] = np.mean(mask)
    return out
# def box(img, r):
#     """ O(1) box filter
#         img - >= 2d image
#         r   - radius of box filter
#     """
#     (rows, cols) = img.shape[:2]
#     imDst = np.zeros_like(img)
#
#     tile = [1] * img.ndim   #[1,1,..,1] hai chieu thi img.ndim la 2 = [1,1]
#
#     tile[0] = r
#     # print("img",img)
#     imCum = np.cumsum(img, 0)   #1200,800
#     imDst[0:r+1, :, ...] = imCum[r:2*r+1, :, ...]
#     imDst[r+1:rows-r, :, ...] = imCum[2*r+1:rows, :, ...] - imCum[0:rows-2*r-1, :, ...]
#     imDst[rows-r:rows, :, ...] = np.tile(imCum[rows-1:rows, :, ...], tile) - imCum[rows-2*r-1:rows-r-1, :, ...]
#
#     tile = [1] * img.ndim
#     tile[1] = r
#     imCum = np.cumsum(imDst, 1)
#     imDst[:, 0:r+1, ...] = imCum[:, r:2*r+1, ...]
#     imDst[:, r+1:cols-r, ...] = imCum[:, 2*r+1 : cols, ...] - imCum[:, 0 : cols-2*r-1, ...]
#     imDst[:, cols-r: cols, ...] = np.tile(imCum[:, cols-1:cols, ...], tile) - imCum[:, cols-2*r-1 : cols-r-1, ...]
#
#     return imDst

def mean_of_all_guidance_at_pixel_k(i,j,local_var_I,local_var_I1):
    return (local_var_I[i,j]+local_var_I1[i,j])/2
def edge_aware_constraint(i,j,local_var_I,local_var_I1,covI1p,covIp):
    global edge_aware_mu
    edge_aware_mu = np.array([])
    t = local_var_I[i,j]/np.mean(local_var_I)
    edge1 = (2/(1+math.exp(-t)) - 1)
    if covIp[i,j] > 0 :
        edge_aware_mu = np.append(edge_aware_mu,edge1*1)
    else :
        edge_aware_mu = np.append(edge_aware_mu,edge1*(-1))

    t = local_var_I1[i,j]/np.mean(local_var_I1)
    edge1 = (2/(1+math.exp(-t)) - 1)
    if covI1p[i,j] > 0 :
        edge_aware_mu = np.append(edge_aware_mu, edge1 * 1)
    else:
        edge_aware_mu = np.append(edge_aware_mu, edge1 * (-1))

    return edge_aware_mu;
def edge_aware(i,j,eps,local_var_I,local_var_I1):
    e = (0.001*256)**2
    wk1 = (eps * np.mean(local_var_I) + eps * mean_of_all_guidance_at_pixel_k(i,j,local_var_I,local_var_I1)) / (local_var_I[i,j]+e)
    wk2 = (eps * np.mean(local_var_I1) + eps * mean_of_all_guidance_at_pixel_k(i,j,local_var_I,local_var_I1)) / (local_var_I1[i,j]+e)
    Wk = np.array([wk1,wk2]);
    return Wk;

def _gf_gray_multi(I,I1,p,r,eps):
    (rows, cols) = I.shape

    meanI = local_mean(I,r)
    meanI1 = local_mean(I1,r)
    meanII = local_mean(I * I,r)
    meanI1I1 = local_mean(I1*I1,r)
    meanII1 = local_mean(I*I1,r)
    meanIp = local_mean(I*p,r)
    meanI1p = local_mean(I1*p,r)
    meanp = local_mean(p,r)
    # phuong sai kenh 1
    local_varianceI = meanII - meanI*meanI
    # phuong sai kenh 2
    local_varianceI1 = meanI1I1 - meanI1*meanI1
    # hiep phuong sai giua hai channel
    covII = meanII - meanI*meanI
    covII1 = meanII1 - meanI*meanI1
    covI1I = meanII1 - meanI*meanI1
    covI1I1 = meanI1I1 - meanI1*meanI1
    # hiep phuong sai kenh va inputimage
    covI1p = meanI1p - meanI1 * meanp
    covIp = meanIp - meanI * meanp
    #tao 2 mang chua akj ung voi 2 kenh
    Akk = np.zeros((rows,cols,2))
    for i in range(rows):
        for j in range(cols):
            W = np.diag(edge_aware(i,j,eps,local_varianceI,local_varianceI1))
            Cj1j2 = np.array([ [covII[i,j],covI1I[i,j]] , [covII1[i,j],covI1I1[i,j]] ]);
            edge_aware_const = edge_aware(i,j,eps,local_varianceI,local_varianceI1) * edge_aware_constraint(i,j,local_varianceI,local_varianceI1,covI1p,covIp)
            Cj0 = np.array([covIp[i,j],covI1p[i,j]])

            Ak =np.dot(np.linalg.inv(Cj1j2 + W),(Cj0 + edge_aware_const).T)
            Akk[i,j,0] = Ak[0]
            Akk[i,j,1] = Ak[1]
    print(Akk[:, :, 0])

    bk = meanp - Akk[:,:,0]* meanI - Akk[:,:,1]* meanI1
    print(bk);

    q = local_mean((Akk[:,:,0] * I),r) + local_mean((Akk[:,:,1] * I1),r) + local_mean(bk,r)
    return q

def test_gf():
    import imageio
    # cat = imageio.imread('cat.bmp').astype(np.float32) / 255
    img1 = cv2.imread('img1_1.png').astype(np.float32) / 255
    img2 = cv2.imread('img2_2.png').astype(np.float32) / 255
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    # imgh = Image.fromarray((img2*255).astype(np.uint8))
    # imgh.show()
    # imgh.save('xaminnearfocus.png')
    # img = Image.fromarray((img1*255).astype(np.uint8))
    # img.show()
    # img.save('xaminfarfocus.png')
    # imgresult = Image.open('hello.png')
    # imgresult.show()
    # print(cat.shape)
    # print(img1);
    r = 8
    eps = 0.05
    result = _gf_gray_multi(img1,img2,img1,r,eps)
    print(result)
    image_result = Image.fromarray((result * 255).astype(np.uint8))
    image_result.save('hello1.png')
    # print(result)

test_gf();

#anh dau vao
# [[0.5369098  0.5408314  0.5408314  ... 0.04757255 0.03047843 0.07361569]
#  [0.5369098  0.5408314  0.5369098  ... 0.03972941 0.02655686 0.06185098]
#  [0.5369098  0.53298825 0.5369098  ... 0.04757255 0.03188628 0.03580784]
#  ...
#  [0.93346274 0.93346274 0.9371059  ... 0.8937333  0.88981175 0.8858902 ]
#  [0.9294981  0.9294981  0.92169803 ... 0.88981175 0.8937333  0.8858902 ]
#  [0.92557645 0.92557645 0.9256196  ... 0.88981175 0.88981175 0.88981175]]

# Akk[:,:,0]
# [[0.59961164 0.59701911 0.59474473 ... 0.10287588 0.10196289 0.09665193]
#  [0.5971061  0.59460751 0.59258943 ... 0.1249456  0.12488733 0.12060816]
#  [0.59476424 0.59254595 0.59090671 ... 0.14187706 0.14228637 0.13909679]
#  ...
#  [0.55177162 0.5500895  0.54890937 ... 0.5547286  0.55583326 0.55756631]
#  [0.55352371 0.5515031  0.54997432 ... 0.55585676 0.55732084 0.55939498]
#  [0.5559634  0.55361921 0.55176126 ... 0.55762613 0.55941533 0.56181611]]


# bk
# [[-0.02753051 -0.02975316 -0.03198472 ...  0.01280841  0.0120604
#    0.01105689]
#  [-0.02976105 -0.03223834 -0.03476594 ...  0.01463544  0.01380866
#    0.01271779]
#  [-0.03198329 -0.03475817 -0.0376357  ...  0.01635958  0.01539277
#    0.01421332]
#  ...
#  [-0.02897267 -0.0311476  -0.0334744  ... -0.03423702 -0.03183833
#   -0.02959466]
#  [-0.02726674 -0.02915667 -0.0311376  ... -0.03182635 -0.02978004
#   -0.02782682]
#  [-0.0256134  -0.02726489 -0.02896178 ... -0.02957557 -0.02781871
#   -0.02610916]]


# anh sau khi loc
# [[0.60381245 0.60134708 0.5946016  ... 0.0221784  0.01893951 0.02200228]
#  [0.59886125 0.59874195 0.59204692 ... 0.02521229 0.02241799 0.02428203]
#  [0.60159571 0.59217376 0.59016615 ... 0.02787469 0.02434605 0.02375988]
#  ...
#  [0.9895026  0.98461157 0.98246079 ... 0.94053041 0.94478973 0.94756716]
#  [0.99235863 0.98721712 0.97819273 ... 0.94242706 0.94917031 0.94997306]
#  [0.99532827 0.98989958 0.98704346 ... 0.94706253 0.95186234 0.95946194]]

