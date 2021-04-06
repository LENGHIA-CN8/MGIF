import numpy as np

def local_mean(arr,r):
    (rows,cols) = arr.shape[:2];
    out = np.zeros((rows,cols))

    # truyen cols vaf rows 0 vao
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
    #xoa cols va rows 0
    # for i in range(r):
    #     arr = np.delete(arr,0,axis=0)
    #     arr = np.delete(arr,arr.shape[0]-1,axis=0)
    # for i in range(r):
    #     arr = np.delete(arr,0,axis=1)
    #     arr = np.delete(arr,arr.shape[1]-1,axis=1)
    return out
def window(img, r):
    (rows, cols) = img.shape[:2]
    imDst = np.zeros_like(img)

    tile = [1] * img.ndim   #[1,1,..,1] hai chieu thi img.ndim la 2 = [1,1]

    tile[0] = r
    # print("img",img)
    imCum = np.cumsum(img, 0)   #1200,800
    imDst[0:r+1, :] = imCum[r:2*r+1, :]
    imDst[r+1:rows-r, :, ...] = imCum[2*r+1:rows, :, ...] - imCum[0:rows-2*r-1, :, ...]
    imDst[rows-r:rows, :, ...] = np.tile(imCum[rows-1:rows, :, ...], tile) - imCum[rows-2*r-1:rows-r-1, :, ...]

    tile = [1] * img.ndim
    tile[1] = r
    imCum = np.cumsum(imDst, 1)
    imDst[:, 0:r+1, ...] = imCum[:, r:2*r+1, ...]
    imDst[:, r+1:cols-r, ...] = imCum[:, 2*r+1 : cols, ...] - imCum[:, 0 : cols-2*r-1, ...]
    imDst[:, cols-r: cols, ...] = np.tile(imCum[:, cols-1:cols, ...], tile) - imCum[:, cols-2*r-1 : cols-r-1, ...]

    return imDst
def mean(arr, r):
    (rows, cols) = arr.shape[:2];
    out = np.zeros((rows, cols))
    N = window(np.ones([rows, cols]), r)
    # mask = np.zeros((2*r+1,2*r+1))
    # truyen cols vaf rows 0 vao anh
    zeros_1 = np.zeros((1, arr.shape[1]))
    for i in range(r):
        arr = np.insert(arr, arr.shape[0], zeros_1, axis=0)
        arr = np.insert(arr, 0, zeros_1, axis=0)
    zeros_2 = np.zeros((1, arr.shape[0]))
    for i in range(r):
        arr = np.insert(arr, arr.shape[1], zeros_2, axis=1)
        arr = np.insert(arr, 0, zeros_2, axis=1)
    for i in range(arr.shape[0] - (2 * r)):
        for j in range(arr.shape[1] - (2 * r)):
            # print(j)
            mask = arr[i:i + 2 * r + 1, j:j + 2 * r + 1]
            out[i, j] = np.sum(mask) / N[i, j]
    return out
def _gf_gray(I, p, r, eps):

    meanI = local_mean(I,r)
    meanP = local_mean(p,r)
    corrI = local_mean(I*I,r)
    corrIp = local_mean(I*p,r)
    varI = corrI - meanI * meanI
    covIp = corrIp - meanI * meanP


    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = mean(a,r)
    meanB = mean(b,r)

    q = meanA * I + meanB
    return q

def test_gf():
    import cv2
    import imageio
    cat = imageio.imread('./input_image/cat.bmp').astype(np.float32) / 255
    # img1 = cv2.imread('img1_1.png').astype(np.float32)
    # img2 = cv2.imread('img2_2.png').astype(np.float32)
    # img1 = cv2.resize(img1, (800, 600))
    # img2 = cv2.resize(img2, (800, 600))
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    r = 8
    eps = 0.05
    result = _gf_gray(cat,cat,r,eps)
    print('Result',result)
    # print(np.max(result))
    # cv2.imwrite('AnhResultGIF_anhcuatoi.png', result)
    # imageio.imwrite('cat_showsmoothed1.png', result)
    cv2.imshow('cat_showsmoothed1.png',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
test_gf();



