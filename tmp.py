import numpy as np
import cv2


x=np.array([1,2])
y=np.array([3,4])
print(x+y)

img = np.zeros([500,500],dtype="float32")
img2 = np.zeros([500,500],dtype="float32")
img3 = np.zeros([500,500],dtype="float32")
img4 = np.zeros([500,500],dtype="uint8")

for i in range(500):
    for j in range(500):
        img[i][j]=(np.sin(0.5*j+25)+np.sin(0.5*i+25))/2

for i in range(5):
    for j in range(5):
        if((i+j)%2==0):
            for k in range(100):
                for l in range(100):
                    img4[i*100+k][j*100+l]=255

img5=cv2.imread('C:\\Users\\user\\pythontmp\\hus.jpg',-1)
#(b, g, r)=cv2.split(img5)
#img5=cv2.merge([r,g,b])
color=np.zeros((415,600,3),dtype="uint8")
color[:,:,1]=img5[:,:,1]
cv2.imshow("img5",img5)
cv2.imshow("img_B",color)
#cv2.imshow("img_G",g)
#cv2.imshow("img_R",r)
#cv2.imshow("t",img)
#cv2.imshow("e",img2)
#cv2.imshow("s",img3)
#cv2.imshow("test",img4)
cv2.waitKey(0)
cv2.destroyAllWindows()

