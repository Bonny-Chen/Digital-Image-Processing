import numpy as np
import cv2
import matplotlib.pyplot as plt

origin=cv2.imread('C:\\Users\\user\\pythontmp\\g.jpg',-1)

# Read image and change it to gray
gray = cv2.imread('C:\\Users\\user\\pythontmp\\g.jpg',0)
# Black and white histogram
plt.hist(gray.ravel(),256,[0,256])
plt.show()

def histequ(image):
    row,column = image.shape
    # Array[256] count color appear times
    arr = np.zeros(shape=(256,1)) 
    for i in range(row):
        for j in range(column):
            if(image[i][j]):
                arr[image[i][j]]=arr[image[i][j]]+1

    # Accumulate
    cdf=arr.cumsum()
    # Remove zero in histogram
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    equ = cdf[image]

    plt.hist(equ.ravel(), 256, [0, 256])
    plt.show()
    return equ

histequ(gray)
# Split
b, g, r = cv2.split(origin)
plt.hist(b.ravel(), 256, [0, 256],color="b")      #Blue
plt.show()
plt.hist(g.ravel(), 256, [0, 256],color="g")      #Green
plt.show()
plt.hist(r.ravel(), 256, [0, 256],color="r")      #Red
plt.show()
be=histequ(b)
ge=histequ(g)
re=histequ(r)

merged = cv2.merge([be, ge, re])
cv2.imshow("Merged", merged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imshow("equ",equ)
# # cv2.imshow('Black White', gray)
# # cv2.imshow('origin', origin)
# cv2.imshow("b", b)
# cv2.imshow("g", g)
# cv2.imshow("r", r)
# cv2.waitKey(0)



