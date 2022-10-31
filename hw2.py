import cv2
import numpy as np
# Erosion
def Erosion(img):
    row = img.shape[0]
    column = img.shape[1]
    tmp=np.zeros((row,column),np.float32)
    for i in range (1,row-2):
        for j in range (1,column-2):
            min = img[i,j]
            for k in range (i-1,i+3):
                for l in range (j-1,j+3):
                    if k<0|k>=row-1|l<0|l>=column-1:
                        continue
                    if img[k,l]<min:
                        min=img[k,l]
            tmp[i,j]=min
    return tmp
# Dilation
def Dilation(img):
    row = img.shape[0]
    column = img.shape[1]
    tmp=np.zeros((row,column),np.uint8)
    for i in range (1,row-2):
        for j in range (1,column-2):
            max=img[i,j]
            for k in range (i-1,i+3):
                for l in range (j-1,j+3):
                    if k<0|k>=row-1|l<0|l>=column-1:
                        continue
                    if img[k,l]>max:
                       max=img[k,l]
            tmp[i,j]=max
    return tmp

image1 = cv2.imread('C:\\Users\\user\\pythontmp\\1.jpg')
image2 = cv2.imread('C:\\Users\\user\\pythontmp\\2.jpg')
image = cv2.imread('C:\\Users\\user\\pythontmp\\3.jpg')
#change to HSV and rgb
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#find skin color
lower = np.array([0, 10, 60])
upper = np.array([30, 180, 255])
mask = cv2.inRange(hsv, lower, upper)
skin = cv2.bitwise_and(image, image, mask=mask)
mask1 = cv2.inRange(rgb, lower, upper)
skin1 = cv2.bitwise_and(image, image, mask=mask)
# cv2.imshow("skinHSV.jpg", skin)
# cv2.imshow("skinRGB.jpg", skin1)
#change to gray then to binary
h,s,v = cv2.split(skin)
ret, thresh = cv2.threshold(v, 127, 255, cv2.THRESH_BINARY)
grayimg1 = cv2.cvtColor(skin1, cv2.COLOR_BGR2GRAY)
ret1, thresh1 = cv2.threshold(grayimg1, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow("1.jpg", thresh)
# cv2.imshow("2.jpg", thresh1)
# hsv
ero=Erosion(thresh)   
dila=Dilation(thresh)  
opening = Dilation(ero)
closing= Erosion(dila)  
# rgb
ero1=Erosion(thresh1)   
dila1=Dilation(thresh1) 
opening1 = Dilation(ero1)
closing1= Erosion(dila1)  
# cv2.imshow("binary.jpg", thresh)
# cv2.imshow("binary.jpg", thresh1)
# cv2.imshow("erosin.jpg", ero)
# cv2.imshow("dilasion.jpg", dila)
# cv2.imshow("erosin1.jpg", ero1)
# cv2.imshow("dilasion1.jpg", dila1)

cv2.imshow("openHSV.jpg", opening)
cv2.imshow("closeHSV.jpg", closing)
cv2.imshow("openRGB.jpg", opening1)
cv2.imshow("closeRGB.jpg", closing1)


cv2.waitKey(0)