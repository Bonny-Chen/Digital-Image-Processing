import cv2
import numpy as np

def canny(img):
     
    # GaussianBlur
    img = cv2.GaussianBlur(img,(5,5),0)
    rows,cols = img.shape
    # grad = np.zeros(img.shape,np.float32)
    # theta = np.zeros(img.shape,np.float32)

    # Sobel
    sobelx = cv2.Sobel(np.float32(img),cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(np.float32(img),cv2.CV_64F,0,1,ksize=3)

    # sobelx = cv2.convertScaleAbs(sobelx)             # 取絕對值
    # sobely = cv2.convertScaleAbs(sobely)             # 取絕對值
    # grad, theta = cv2.cartToPolar(sobelx, sobely, angleInDegrees = True) 
        
    grad = np.hypot(sobelx,sobely)      #梯度 sqrt(x*x + y*y)
    theta = np.arctan2(sobelx, sobely)       # direction 
    theta = np.rad2deg(theta)                # 180 * x / pi
    # print(theta)

    # separate to 0° 45° 90° -45°
    for i in range(1,rows - 2):
        for j in range(1, cols - 2):
            if ( theta[i,j] >= -22.5 and theta[i,j]<= 22.5 ) or( theta[i,j] <= -157.5 and theta[i,j] >= -180 ) or( theta[i,j] >= 157.5 and theta[i,j] <= 180 ):
                theta[i,j] = 0.0
            elif ( theta[i,j] >= 22.5 and theta[i,j]< 67.5 ) or( theta[i,j] <= -112.5 and theta[i,j] > -157.5 ):
                theta[i,j] = 45.0
            elif ( theta[i,j] >= 67.5 and theta[i,j]< 112.5 ) or( theta[i,j] <= -67.5 and theta[i,j] > -112.5 ):
                theta[i,j] = 90.0
            elif ( theta[i,j] >= 112.5 and theta[i,j]< 157.5 ) or( theta[i,j] <= -22.5 and theta[i,j] > -67.5 ):
                theta[i,j] = -45.0

    # NMS 3*3 方向上最大
    NMS = np.zeros(grad.shape)
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            if (theta[i,j] == 0.0) and (grad[i,j] == np.max([grad[i,j],grad[i+1,j],grad[i-1,j]]) ):
                    NMS[i,j] = grad[i,j]

            if (theta[i,j] == -45.0) and grad[i,j] == np.max([grad[i,j],grad[i-1,j-1],grad[i+1,j+1]]):
                    NMS[i,j] = grad[i,j]

            if (theta[i,j] == 90.0) and  grad[i,j] == np.max([grad[i,j],grad[i,j+1],grad[i,j-1]]):
                    NMS[i,j] = grad[i,j]

            if (theta[i,j] == 45.0) and grad[i,j] == np.max([grad[i,j],grad[i-1,j+1],grad[i+1,j-1]]):
                    NMS[i,j] = grad[i,j]

    # TL/TH and eight connections
    result = np.zeros(NMS.shape)
    TL = 50
    TH = 100
    for i in range(1,rows-1): 
        for j in range(1,cols-1):
            if NMS[i,j] < TL:
                result[i,j] = 0
            elif NMS[i,j] > TH:
                result[i,j] = 255
            elif NMS[i+1,j] < TH or NMS[i-1,j] < TH or NMS[i,j+1] < TH or NMS[i,j-1] < TH or NMS[i-1, j-1] < TH or  NMS[i-1, j+1] < TH or NMS[i+1, j+1] < TH  or  NMS[i+1, j-1] < TH:
                result[i,j] = 255
    return result

img1 = cv2.imread('C:\\Users\\user\\pythontmp\\tae.jpg')
#house
img1 = cv2.resize(img1, (392, 512), interpolation=cv2.INTER_AREA)
#church 
# img1 = cv2.resize(img1, (512, 392), interpolation=cv2.INTER_AREA) 
# people
# img1 = cv2.resize(img1, (392, 512), interpolation=cv2.INTER_AREA)
sign = cv2.imread('C:\\Users\\user\\pythontmp\\b.jpg')
sign = cv2.resize(sign, (50, 50), interpolation=cv2.INTER_AREA)
rows,cols,channels = sign.shape
roi = img1[0:rows, 0:cols ]
signgray = cv2.cvtColor(sign,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(signgray, 175, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)   # 反位運算
img1_bg = cv2.bitwise_and(roi,roi,mask = mask)  # 在img1上面，將sign和mask and 使值為0
sign_fg = cv2.bitwise_and(sign,sign,mask = mask_inv)    # 取 roi 中與 mask_inv 中不為零的值對應的畫素的值，其他值為 0
dst = cv2.add(img1_bg,sign_fg)
img1[0:rows, 0:cols ] = dst
cv2.imwrite('output.jpg', img1)


img = cv2.imread("output.jpg")
# img = cv2.resize(img, (392, 512), interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resCanny = canny(img)
cv2.imshow("canny",resCanny)   
edges = cv2.Canny(img,50,150,apertureSize = 3)
# cv2.imshow("d",edges)


# Hough Transform
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=50,maxLineGap=10)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(resCanny, (x1, y1), (x2, y2), (255, 255, 255), 2)
# Show result
# cv2.imwrite('output.jpg', resCanny)
cv2.imshow("Hough transform",resCanny)
# print(resCanny.shape)


cv2.waitKey(0)
cv2.destroyAllWindows()