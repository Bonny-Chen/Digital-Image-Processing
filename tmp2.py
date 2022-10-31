import cv2
import numpy as np
import random
from scipy.spatial import distance

def resizeImage(img):
    img = cv2.imread(img)
    img = cv2.resize(img,(400,400),interpolation=cv2.INTER_AREA)
    return img

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

img1 = resizeImage('p1.jpg')
img2 = resizeImage('p2.jpg')
img3 = resizeImage('p3.jpg')

orb = cv2.ORB_create()
kp1 , des1 = orb.detectAndCompute(img1,None)
kp2 , des2 = orb.detectAndCompute(img2,None)
kp3 , des3 = orb.detectAndCompute(img3,None)

matches1 = findMatch(kp1, des1, img1, kp2, des2, img2, 0.6)
matches2 = findMatch(kp1, des1, img1, kp3, des3, img3, 0.6)

# print(matches1)

def knn(a,keypointlength,threshold):
    #threshold=0.2
    bestmatch=np.zeros((keypointlength),dtype= np.int8)
    img1index=np.zeros((keypointlength),dtype=np.int8)
    distance=np.zeros((keypointlength))
    index=0
    verybest =[]
    for j in range(keypointlength):
        new=a[j]
        old=new.tolist()
        new.sort()
        minval1=new[0]    # min 
        # print("min",minval1)
        minval2=new[1]    # second min
        itemindex1 = old.index(minval1)   #index 
        itemindex2 = old.index(minval2)          

        if minval1<threshold*minval2: 
            bestmatch[index]=itemindex1
            distance[index]=minval1
            img1index[index]=j
            index=index+1
            verybest.append([minval1+minval2])
    return verybest

distance = distance.cdist(des1,des2,metric='euclidean') 
matchestmp = knn(distance,min(len(des1),len(des2)),0.5)
# print("matches",len(matchestmp))
# print(matches1)
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
    # matrix = matrix/matrix[2,2]
    # print(matrix)
    # ori_point=np.float32(ori_point)
    # dst_point = np.float32(dst_point)
    return matrix #,ori_point,dst_point

def Inlier(matches, H):
    line1 = np.concatenate((matches[:,0:2], np.ones((len(matches), 1))), axis=1)
    # print('1',line1)
    line2 = matches[:,2:4]
    # print('2',line2)
    linetmp = np.zeros((len(matches), 2))
    for i in range(len(matches)):
        matrix = np.dot(H, line1[i])
        linetmp[i] = (matrix/matrix[2])[0:2]
    outlier = np.linalg.norm(line2 - linetmp , axis=1)**2
    return outlier

def Ransac(matches, threshold, iteration):
    nBest = 0
    for i in range(iteration):
        # print(len(matches))
        index = random.sample(range(len(matches)), 5)   #random 4 index
        points = [matches[i] for i in index]            #random points in matches
        H= PerspectiveTransform(points)
        outlier = Inlier(matches, H)
        # print(outlier)
        index = np.where(outlier < threshold)[0]
        inliers = matches[index]
        # print(inliers)
        if len(inliers) > nBest:
            best_inliers = inliers.copy()
            nBest = len(inliers)
            best_H = H.copy()
            # b_ori = ori_point.copy()
            # b_dst = dst_point.copy()
            
    print("inliers/matches: {}/{}".format(nBest, len(matches)))
    return best_H   #,b_ori,b_dst

H1= Ransac(matches1, 0.5, 2000)
H2= Ransac(matches1, 0.5, 2000)
# print(b_dst1)
# b_dst1 = np.reshape(b_dst1,(4,2))
# b_dst2 = np.reshape(b_dst2,(4,2))

# print(b_dst1)
# print(b_dst1.shape)

def Stitch(img1,img2,matrix,alpha):
    rows,cols=img2.shape[:2]
    result = np.zeros((max(img1.shape[0],img2.shape[0]), cols*2,3),np.uint8)
    matrix=np.mat(matrix)
    RANGE = 100
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            point = np.array([[j], [i], [1]])
            point = np.mat(point)
            x, y, z = np.dot(matrix,point)
            x = x/z 
            y = y/z
            if x < 0 or x >= img2.shape[1] or y<0 or y >= img2.shape[0]:    #超過範圍不給值
                result[i,j,:] = 0
                continue
            if result[i,j,0]!=0 or result[i,j,1]!=0 or result[i,j,2]!=0:    #already have value
                continue
            if j<=RANGE:
                result[i,j,:] = img2[int(y),int(x),:]*(1-alpha)     # change weight
            else :
                result[i,j,:] = img2[int(y),int(x),:]
    
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]): 
            if result[i,j,0]==0 and result[i,j,1]==0 and result[i,j,2]==0:
                result[i,j,:]= img1[i,j,:]
            elif j <= RANGE :
                result[i,j,:]= img1[i,j,:]*alpha + result[i,j,:]    # change weight
    return result

result1 = cv2.imshow("twoPic",Stitch(img1, img2, H1,0.7))
cv2.imwrite('twoPic.jpg',Stitch(img1, img2, H1,0.7))
twoPic = cv2.imread('twoPic.jpg')
result2 = cv2.imshow("threePic",Stitch(twoPic, img3, H2,0.7))
cv2.waitKey()