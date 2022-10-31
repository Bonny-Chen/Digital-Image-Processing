import cv2
import numpy as np

img = cv2.imread('C:\\Users\\user\\pythontmp\\g2.jpg')
img = cv2.resize(img, (512, 392), interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)

cv2.imshow("d",img)
cv2.imwrite('houghlines5.jpg',img)
cv2.waitKey(0)
# import cv2
# import numpy as np
# # 載入影象
# img1 = cv2.imread('C:\\Users\\user\\pythontmp\\g1.jpg')
# img1 = cv2.resize(img1, (392, 512), interpolation=cv2.INTER_AREA)
# img2 = cv2.imread('C:\\Users\\user\\pythontmp\\b.jpg')
# img2 = cv2.resize(img2, (50, 100), interpolation=cv2.INTER_AREA)
# rows,cols,channels = img2.shape
# roi = img1[0:rows, 0:cols ]

# img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)      # 將圖片灰度化
# ret, mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)#ret是閾值（175）mask是二值化影象
# mask_inv = cv2.bitwise_not(mask)#獲取把logo的區域取反 按位運算

# img1_bg = cv2.bitwise_and(roi,roi,mask = mask)#在img1上面，將logo區域和mask取與使值為0

# # 取 roi 中與 mask_inv 中不為零的值對應的畫素的值，其他值為 0 。
# # 把logo放到圖片當中
# img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)#獲取logo的畫素資訊

# dst = cv2.add(img1_bg,img2_fg)#相加即可
# img1[0:rows, 0:cols ] = dst
# cv2.imshow('res',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# img = cv2.imread('C:\\Users\\user\\pythontmp\\g1.jpg')
# img = cv2.resize(img, (392, 512), interpolation=cv2.INTER_AREA)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,50,150,apertureSize = 3)
# lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
# for line in lines:
#     x1,y1,x2,y2 = line[0]
#     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

# cv2.imshow("Result Image", img)
# cv2.waitKey(0)

# import numpy as np 
# import os 
# import cv2 
# import matplotlib.pyplot as plt 
  
   
# # defining the canny detector function 
   
# # here weak_th and strong_th are thresholds for 
# # double thresholding step 
# def Canny_detector(img, weak_th = None, strong_th = None): 
      
#     # conversion of image to grayscale 
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
       
#     # Noise reduction step 
#     img = cv2.GaussianBlur(img, (5, 5), 1.4) 
       
#     # Calculating the gradients 
#     gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3) 
#     gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3) 
      
#     # Conversion of Cartesian coordinates to polar  
#     mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True) 
#     print(ang)
#     # setting the minimum and maximum thresholds  
#     # for double thresholding 
#     mag_max = np.max(mag) 
#     if not weak_th:weak_th = mag_max * 0.1
#     if not strong_th:strong_th = mag_max * 0.5
      
#     # getting the dimensions of the input image   
#     height, width = img.shape 
       
#     # Looping through every pixel of the grayscale  
#     # image 
#     for i_x in range(width): 
#         for i_y in range(height): 
               
#             grad_ang = ang[i_y, i_x] 
#             grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang) 
               
#             # selecting the neighbours of the target pixel 
#             # according to the gradient direction 
#             # In the x axis direction 
#             if grad_ang<= 22.5: 
#                 neighb_1_x, neighb_1_y = i_x-1, i_y 
#                 neighb_2_x, neighb_2_y = i_x + 1, i_y 
              
#             # top right (diagnol-1) direction 
#             elif grad_ang>22.5 and grad_ang<=(22.5 + 45): 
#                 neighb_1_x, neighb_1_y = i_x-1, i_y-1
#                 neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
              
#             # In y-axis direction 
#             elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90): 
#                 neighb_1_x, neighb_1_y = i_x, i_y-1
#                 neighb_2_x, neighb_2_y = i_x, i_y + 1
              
#             # top left (diagnol-2) direction 
#             elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135): 
#                 neighb_1_x, neighb_1_y = i_x-1, i_y + 1
#                 neighb_2_x, neighb_2_y = i_x + 1, i_y-1
              
#             # Now it restarts the cycle 
#             elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180): 
#                 neighb_1_x, neighb_1_y = i_x-1, i_y 
#                 neighb_2_x, neighb_2_y = i_x + 1, i_y 
               
#             # Non-maximum suppression step 
#             if width>neighb_1_x>= 0 and height>neighb_1_y>= 0: 
#                 if mag[i_y, i_x]<mag[neighb_1_y, neighb_1_x]: 
#                     mag[i_y, i_x]= 0
#                     continue
   
#             if width>neighb_2_x>= 0 and height>neighb_2_y>= 0: 
#                 if mag[i_y, i_x]<mag[neighb_2_y, neighb_2_x]: 
#                     mag[i_y, i_x]= 0
   
#     weak_ids = np.zeros_like(img) 
#     strong_ids = np.zeros_like(img)               
#     ids = np.zeros_like(img) 
       
#     # double thresholding step 
#     for i_x in range(width): 
#         for i_y in range(height): 
              
#             grad_mag = mag[i_y, i_x] 
              
#             if grad_mag<weak_th: 
#                 mag[i_y, i_x]= 0
#             elif strong_th>grad_mag>= weak_th: 
#                 ids[i_y, i_x]= 1
#             else: 
#                 ids[i_y, i_x]= 2
       
       
#     # finally returning the magnitude of 
#     # gradients of edges 
#     return mag 
   
# img = cv2.imread('C:\\Users\\user\\pythontmp\\g2.jpg') 
# frame= cv2.resize(img, (392, 512), interpolation=cv2.INTER_AREA)
# # calling the designed function for 
# # finding edges 
# canny_img = Canny_detector(frame) 
