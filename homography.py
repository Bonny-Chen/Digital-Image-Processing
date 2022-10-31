import cv2
import numpy as np

# Mouse Listener
clicks = list() 
def PointAndClick(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONUP :
        if len(clicks)<4:
            clicks.append([x, y])
        print(clicks)
        return clicks
        
img_target = cv2.imread('target.jpg')
rows,cols,color=img_target.shape
# trans_img=cv2.resize(img2,(cols,rows),interpolation=cv2.INTER_AREA)#####
cv2.imshow('Point-and-Click',img_target)
cv2.setMouseCallback('Point-and-Click',PointAndClick,clicks)
cv2.waitKey()

ori_point = np.float32([[0,0],[cols,0],[cols,rows],[0,rows]]) #順時針
dst_point = np.float32(clicks)

def PerspectiveTransform(ori_point,dst_point):
    A = np.zeros((8, 9))
    X = np.zeros((9, 1))
    for i in range(4):
        A_i = ori_point[i]
        X_i = dst_point[i]
        A[i*2,:] = A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0]*X_i[0], -A_i[1]*X_i[0], -X_i[0]
        A[i*2+1,:] = 0, 0, 0, A_i[0], A_i[1], 1, -A_i[0]*X_i[1], -A_i[1]*X_i[1], -X_i[1]
        # X[i*2] = X_i[0] #0,2,4,6
        # X[i*2+1] = X_i[1] #1,3,5,7
    # A = np.mat(A)
    # matrix = A.I * X #inverse
    # matrix=np.append(matrix,[[1]],axis=0)   #matrix[3][3]=1
    # matrix = matrix.reshape((3, 3))
    U, s, V = np.linalg.svd(A)
    # matrix = np.eye(3)
    matrix = V[-1].reshape(3,3)
    # matrix = matrix/matrix[2,2]
    # print(matrix)
    return matrix
matrix = PerspectiveTransform(ori_point, dst_point)
# matrix = cv2.getPerspectiveTransform(ori_point,dst_point)

def DestinationScan(img,rows,cols,dst,matrix):
    img =cv2.resize(img,(cols,rows),interpolation=cv2.INTER_AREA)   #調整與target_img相同大小
    Hmin=int(min(dst[:,1])) #點選範圍做轉換
    Wmin=int(min(dst[:,0]))
    Hmax=int(max(dst[:,1]))
    Wmax=int(max(dst[:,0]))
    result = np.zeros((img.shape[0], img.shape[1],3),np.uint8)
    matrix=np.mat(matrix)
    inMatrix=matrix.I
    for i in range(Hmin,Hmax):
        for j in range(Wmin,Wmax):
            point = np.array([[j], [i], [1]])
            point = np.mat(point)
            x, y, z = np.dot(inMatrix,point)
            x = x/z
            y = y/z
            if x < 0 or x >= img.shape[1] or y<0 or y >= img.shape[0]: #超過範圍不給值
                # result[i,j,:] = 0
                continue
            result[i,j,:] = img[int(y),int(x),:]
    return result

# 挖空選取範圍
cv2.fillConvexPoly(img_target,dst_point.astype(int),(0,0,0))

# 影片轉換
cap = cv2.VideoCapture("puipui.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) #FPS
total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)	#總幀數
print(fps,total_frame)

k=0
while(True):
    cap.set(cv2.CAP_PROP_POS_FRAMES,k)  #設定讀取偵數
    ret, frame = cap.read()
    trans_img = DestinationScan(frame,rows,cols,dst_point,matrix)
    # trans_img = cv2.warpPerspective(frame, matrix, (cols, rows))
    result = img_target+trans_img
    cv2.imshow('Point-and-Click video',result)
    # cv2.imwrite('tmp.jpg',result)
    print(k)
    k=k+50
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break
cap.release()

cv2.waitKey()
cv2.destroyAllWindows()


# # 圖片轉換
# # Find Face
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# img = cv2.imread('tzuyu.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray,
#                                         scaleFactor=1.2,
#                                         minNeighbors=3,)
# for (x,y,w,h) in faces:
#     img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# # print(y,x,y+h,x+w)
# # cv2.imshow('face',img)

# img_special=cv2.imread('special.png')

# def FindBlackVertex(img,left_i,left_j,right_i,right_j):
#     rows,cols,color = img.shape
#     for i in range(rows):
#         for j in range(cols):    
#             if img[i,j,0]==0 and img[i,j,1]==0 and img[i,j,2]==0:
#                 left_i=i
#                 left_j=j
#                 break
#     for i in range(rows-1,0,-1):
#         for j in range(cols-1,0,-1):    
#             if img[i,j,0]==0 and img[i,j,1]==0 and img[i,j,2]==0:
#                 right_i=i
#                 right_j=j
#                 break
#     return left_i,left_j,right_i,right_j

# left_i,left_j,right_i,right_j=FindBlackVertex(img_special,0,0,0,0)
# # print(left_i, left_j)
# # print(right_i,right_j)

# # Resize Special Image
# special_black_height= left_i-right_i
# special_black_width = right_j-left_j

# if(special_black_height>h):
#     size_h = h/special_black_height
# else :
#     size_h = special_black_height/h

# if(special_black_width>w):
#     size_w = w/special_black_width
# else :
#     size_w = special_black_width/w

# rows,cols,color = img_special.shape
# re_special=cv2.resize(img_special,(int(cols*size_w),int(rows*size_h)),interpolation=cv2.INTER_AREA)
# # cv2.imshow("re_special",re_special)

# re_height,re_width,re_color=re_special.shape   # get resized h,w
# # print(re_height,re_width,re_color)
# img_height,img_width,img_color=img.shape    # get origin image h,w
# # print(img_height,img_width,img_color)

# # count resized black point
# right_i=int(right_i*size_h)   
# left_j=int(left_j*size_w)
# row=y-right_i   # re_special 左上角座標
# col=x-left_j
# # 擴成與原圖大小一樣
# re_special = cv2.copyMakeBorder(re_special,row,img_height-re_height-row,col,img_width-re_width-col, cv2.BORDER_CONSTANT,value=[255,255,255])
# # cv2.imshow('re_special',re_special)

# # 合成照片
# hsv = cv2.cvtColor(re_special, cv2.COLOR_BGR2HSV)
# lower = np.array([0, 0, 240])
# upper = np.array([255, 15, 255])
# white_mask = cv2.inRange(hsv, lower, upper)
# not_white=~white_mask
# # cv2.imshow('mask',white_mask)
# # cv2.imshow('ground',not_white)
# lower = np.array([0, 0, 0])
# upper = np.array([180, 255, 200])
# black_mask = cv2.inRange(hsv, lower, upper)

# remain_mask=white_mask+black_mask
# remain=~remain_mask
# # cv2.imshow("remain_mask",remain_mask)

# img2=cv2.imread('tzuyu.jpg')
# composed_img=cv2.bitwise_and(img2, img2, mask = remain_mask)
# # cv2.imshow('composed_img',composed_img)
# re_special=cv2.bitwise_and(re_special, re_special, mask = remain)
# # cv2.imshow('re_special',re_special)
# composed_img = cv2.add(composed_img,re_special)
# cv2.imshow('composed_img',composed_img)
# rows,cols,color = img_target.shape
# # img2=cv2.imread('tzuyu.jpg')
# trans_img = DestinationScan(composed_img,rows,cols,dst_point,matrix,img_target)
# # cv2.imshow('trans_img',trans_img)
# result = img_target+trans_img
# cv2.imshow('Point-and-Click picture',result)
# cv2.waitKey()
# cv2.destroyAllWindows()
