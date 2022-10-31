import cv2
import numpy as np

#read
image = cv2.imread('C:\\Users\\user\\pythontmp\\3.jpg')
b,g,r=cv2.split(image)
# cv2.imshow("b",b)
# cv2.imshow("g",g)
# cv2.imshow("r",r)
# row = image.shape[0]
# column = image.shape[1]
# tmp=np.zeros((row,column),np.uint8)
# for i in range (row):
#     for j in range (column):
        # b,g,r=cv2.split(image[i][j])

# mm=rgb2hsv(r,g,b)
# cv2.imshow("mm",mm)
#change to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#find skin color
lower = np.array([0, 10, 60])
upper = np.array([30, 180, 255])
mask = cv2.inRange(hsv, lower, upper)
skin = cv2.bitwise_and(image, image, mask=mask)

mask1 = cv2.inRange(rgb, lower, upper)
skin1 = cv2.bitwise_and(image, image, mask=mask)
#change to gray then to binary
h, s, v1 = cv2.split(skin)
# cv2.imshow("gray-image",v1)
grayimg1 = cv2.cvtColor(skin1, cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(v1, 127, 255, cv2.THRESH_BINARY)
ret1, thresh1 = cv2.threshold(grayimg1, 127, 255, cv2.THRESH_BINARY)


kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(thresh1,kernel,iterations = 1)
dilation = cv2.dilate(thresh1,kernel,iterations = 1)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

opening1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
closing1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

cv2.imshow("opening.jpg", opening)
cv2.imshow("closing.jpg", closing)
cv2.imshow("opening1.jpg", opening1)
cv2.imshow("closing1.jpg", closing1)
# cv2.imshow("erosin.jpg", erosion)
# cv2.imshow("dilasion.jpg", dilation)
cv2.waitKey(0)