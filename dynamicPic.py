import cv2
import numpy as np
import random
from numba import jit

def findMatch(kp1, des1, img1, kp2, des2, img2, threshold):
    bf = cv2.BFMatcher()
    matche = bf.knnMatch(des1,des2, k=2)
    good = []
    matches = []
    for m,n in matche:
        if m.distance < threshold*n.distance:
            good.append([m])
            matches.append(list(kp1[m.queryIdx].pt + kp2[m.trainIdx].pt))   #origin and des
            # print(list(kp1[m.queryIdx].pt + kp2[m.trainIdx].pt))
    matches = np.array(matches)
    return matches
    
# read
img1 = cv2.imread('source.jpg')
cap = cv2.VideoCapture("puipui.mp4")
total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)	#總幀數

# feature
orb = cv2.SIFT_create()
kp1 , des1 = orb.detectAndCompute(img1,None)
cap.set(cv2.CAP_PROP_POS_FRAMES,450)  #設定讀取偵數
ret, frame = cap.read()
orb = cv2.SIFT_create()
kp2 , des2 = orb.detectAndCompute(frame,None)
matches1 = findMatch(kp1, des1, img1, kp2, des2, frame, 0.6)
# le=0
# for i in range(int(total_frame)):
#     cap.set(cv2.CAP_PROP_POS_FRAMES,i)  #設定讀取偵數
#     ret, frame = cap.read()
#     orb = cv2.SIFT_create()
#     kp2 , des2 = orb.detectAndCompute(frame,None)
#     tmp = findMatch(kp1, des1, img1, kp2, des2, frame, 0.6)
#     if(len(tmp)>le):
#         le=len(tmp)
#         matches1= tmp.copy()
# @jit("float64[:,:](float64[:,:])")
def PerspectiveTransform(points):
    A = np.zeros((8, 9))
    X = np.zeros((9, 1))
    # ori_point=[]
    # dst_point=[]
    for i in range(4):
        A_i = points[i][0:2]
        X_i = points[i][2:4]
        # ori_point.append(A_i)
        # dst_point.append([X_i])
        A[i*2,:] = A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0]*X_i[0], -A_i[1]*X_i[0], -X_i[0]
        A[i*2+1,:] = 0, 0, 0, A_i[0], A_i[1], 1, -A_i[0]*X_i[1], -A_i[1]*X_i[1], -X_i[1]
    U, s, V = np.linalg.svd(A)
    matrix = V[-1].reshape(3,3)
    return matrix

def Inlier(matches, H):
    line1 = np.concatenate((matches[:,0:2], np.ones((len(matches), 1))), axis=1)
    line2 = matches[:,2:4]
    linetmp = np.zeros((len(matches), 2))
    for i in range(len(matches)):
        matrix = np.dot(H, line1[i])
        linetmp[i] = (matrix/matrix[2])[0:2]
    outlier = np.linalg.norm(line2 - linetmp , axis=1)**2
    return outlier

def Ransac(matches, threshold, iteration):
    nBest = 0
    for i in range(iteration):
        index = random.sample(range(len(matches)), 5)   #random 4 index
        points = [matches[i] for i in index]            #random points in matches
        H= PerspectiveTransform(points)
        # print(type(points[0]))
        outlier = Inlier(matches, H)
        index = np.where(outlier < threshold)[0]
        inliers = matches[index]
        if len(inliers) > nBest:
            best_inliers = inliers.copy()
            nBest = len(inliers)
            best_H = H.copy()
            # b_ori = ori_point.copy()
            # b_dst = dst_point.copy()
    print("inliers/matches: {}/{}".format(nBest, len(matches)))
    return best_H

H1= Ransac(matches1, 0.5, 2000)

def DestinationScan(img,img1,rows,cols,matrix):
    print("waiting...")
    result = np.zeros((rows, cols,3),np.uint8)
    matrix=np.mat(matrix)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            point = np.array([[j], [i], [1]])
            point = np.mat(point)
            x, y, z = np.dot(matrix,point)
            x = x/z 
            y = y/z
            if x < 0 or x >= img.shape[1] or y<0 or y >= img.shape[0]:    #超過範圍不給值
                continue
            result[i,j,:] = img[int(y),int(x),:]
    for i in range(result.shape[0]):
        for j in range(result.shape[1]): 
            if result[i,j,0]==0 and result[i,j,1]==0 and result[i,j,2]==0:
                result[i,j,:]= img1[i,j,:]
    return result

rows,cols = img1.shape[:2]  #311,553
k=0
while(True):
    cap.set(cv2.CAP_PROP_POS_FRAMES,k)  #設定讀取偵數
    ret, frame = cap.read()
    trans_img = DestinationScan(frame,img1,rows,cols,H1)
    cv2.imshow('Auto play video',trans_img)
    print(k)
    k=k+10
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break
cap.release()

cv2.waitKey()