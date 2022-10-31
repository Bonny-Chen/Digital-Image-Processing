import cv2
import numpy as np
# Find Face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('tzuyu.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,
                                        scaleFactor=1.2,
                                        minNeighbors=3,)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# print(y,x,y+h,x+w)
# cv2.imshow('face',img)

img_special=cv2.imread('special.png')

def FindBlackVertex(img,left_i,left_j,right_i,right_j):
    rows,cols,color = img.shape
    for i in range(rows):
        for j in range(cols):    
            if img[i,j,0]==0 and img[i,j,1]==0 and img[i,j,2]==0:
                left_i=i
                left_j=j
                break
    for i in range(rows-1,0,-1):
        for j in range(cols-1,0,-1):    
            if img[i,j,0]==0 and img[i,j,1]==0 and img[i,j,2]==0:
                right_i=i
                right_j=j
                break
    return left_i,left_j,right_i,right_j

left_i,left_j,right_i,right_j=FindBlackVertex(img_special,0,0,0,0)
# print(left_i, left_j)
# print(right_i,right_j)

# Resize Special Image
special_black_height= left_i-right_i
special_black_width = right_j-left_j

if(special_black_height>h):
    size_h = h/special_black_height
else :
    size_h = special_black_height/h

if(special_black_width>w):
    size_w = w/special_black_width
else :
    size_w = special_black_width/w

rows,cols,color = img_special.shape
re_special=cv2.resize(img_special,(int(cols*size_w),int(rows*size_h)),interpolation=cv2.INTER_AREA)
# cv2.imshow("re_special",re_special)

re_height,re_width,re_color=re_special.shape   # get resized h,w
# print(re_height,re_width,re_color)
img_height,img_width,img_color=img.shape    # get origin image h,w
# print(img_height,img_width,img_color)

# count resized black point
right_i=int(right_i*size_h)   
left_j=int(left_j*size_w)
row=y-right_i   # re_special 左上角座標
col=x-left_j
# 擴成與原圖大小一樣
re_special = cv2.copyMakeBorder(re_special,row,img_height-re_height-row,col,img_width-re_width-col, cv2.BORDER_CONSTANT,value=[255,255,255])
# cv2.imshow('re_special',re_special)

# 合成照片
hsv = cv2.cvtColor(re_special, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 240])
upper = np.array([255, 15, 255])
white_mask = cv2.inRange(hsv, lower, upper)
not_white=~white_mask
# cv2.imshow('mask',white_mask)
# cv2.imshow('ground',not_white)
lower = np.array([0, 0, 0])
upper = np.array([180, 255, 200])
black_mask = cv2.inRange(hsv, lower, upper)

remain_mask=white_mask+black_mask
remain=~remain_mask
# cv2.imshow("remain_mask",remain_mask)

img2=cv2.imread('tzuyu.jpg')
composed_img=cv2.bitwise_and(img2, img2, mask = remain_mask)
# cv2.imshow('composed_img',composed_img)
re_special=cv2.bitwise_and(re_special, re_special, mask = remain)
# cv2.imshow('re_special',re_special)
composed_img = cv2.add(composed_img,re_special)
cv2.imshow('composed_img',composed_img)
cv2.imwrite('composed.png',composed_img)

# Mouse Listener
clicks = list() 
def PointAndClick(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONUP :
        # cv2.circle(img_target,(x,y),20,(255,0,0),-1)
        if len(clicks)<4:
            clicks.append([x, y])
        print(clicks)
        return clicks
        
img_target = cv2.imread('target.jpg')
rows,cols,color=img_target.shape
composed_img=cv2.resize(composed_img,(cols,rows),interpolation=cv2.INTER_AREA)
cv2.imshow('Point-and-Click',img_target)
cv2.setMouseCallback('Point-and-Click',PointAndClick,clicks)
cv2.waitKey()

# rows,cols,_ = composed_img.shape
points1 = np.float32([[0,0],[cols,0],[cols,rows],[0,rows]]) #順時針
points2 = np.float32(clicks)
matrix = cv2.getPerspectiveTransform(points1,points2)
# 多邊填充
cv2.fillConvexPoly(img_target,points2.astype(int),(0,0,0))
composed_img = cv2.warpPerspective(composed_img, matrix, (cols, rows))
output=img_target+composed_img
cv2.imshow('Point-and-Click Output',output)
cv2.waitKey()
cv2.destroyAllWindows()
