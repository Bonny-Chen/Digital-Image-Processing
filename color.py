import numpy as np
import cv2
import sys
import socket
import time
# sys.path.append('C:/Users/user/Downloads/opencv/sources/samples/python')

# UDP_IP = "140.134.214.152"    #computer
UDP_IP = "10.21.0.47"    #phone
# UDP_IP = "10.21.2.116"  #glasses
UDP_PORT = 5566

print ("UDP IP:", UDP_IP)
print ("UDP port:", UDP_PORT)
sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) # UDP

font = cv2.FONT_HERSHEY_SIMPLEX

# # orange
# low_orange = np.array([10, 100, 100]) 
# upper_orange = np.array([18, 255, 255])
# yellow cream
lower_yellow = np.array([21, 13, 126]) 
upper_yellow = np.array([52, 255, 167])   
# brown chocolate
lower_brown = np.array([0, 89, 19]) 
upper_brown = np.array([22, 255, 255])
# black over
lower_black = np.array([0,0,0])  
upper_black = np.array([180, 255, 46])
# white egg
lower_white = np.array([82,3,95])
upper_white = np.array([175,249,255])

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
while True:
    ret, frame = cap.read()
    frame = cv2.flip( frame, 1 ) 
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv_img,lower_yellow,upper_yellow)
    mask_black = cv2.inRange(hsv_img, lower_black, upper_black)
    mask_white = cv2.inRange(hsv_img, lower_white, upper_white)
    mask_brown = cv2.inRange(hsv_img, lower_brown, upper_brown)
# 篩選顏色

    mask_yellow = cv2.medianBlur(mask_yellow,7)
    mask_black = cv2.medianBlur(mask_black,7)
    mask_white = cv2.medianBlur(mask_white,7)
    mask_brown = cv2.medianBlur(mask_brown,7)
    contours_yellow, hierarchy_yellow = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_black, hierarchy_black = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_white, hierarchy_white = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_brown, hierarchy_brown = cv2.findContours(mask_brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    msg =""

# YELLOW
    flagY=0
    for yellow in contours_yellow:
        (x, y, w, h) = cv2.boundingRect(yellow)
        if w<30 and h<30:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, "Yellow", (x, y - 5), font, 0.7, (0, 255, 255), 2)
        flagY=1
    if(flagY):
        msg = msg+ "cream "

# BROWN
    flagBrown=0
    for brown in contours_brown:
        (x, y, w, h) = cv2.boundingRect(brown)
        if w<30 and h<30:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, "Brown", (x, y - 5), font, 0.7, (0, 255, 255), 2)
        flagBrown=1
    if(flagBrown):
        msg = msg+ "chocolate brown "


# BLACK 
    # flagBK=0
    # for black in contours_black:
    #     (x, y, w, h) = cv2.boundingRect(black)
    #     if w<30 and h<30:
    #         continue
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
    #     cv2.putText(frame, "Black", (x, y - 5), font, 0.7, (0, 0, 0), 2)
    #     flagBK=1
    # if(flagBK):
    #     msg = msg+ "Over Cooked!"
# WHITE
    flagW=0
    for white in contours_white:
        (x, y, w, h) = cv2.boundingRect(white)
        if w<30 and h<30:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, "White", (x, y - 5), font, 0.7, (255, 255, 255), 2)
        flagW=1
    if(flagW):
        msg = msg+ "egg white "

    cv2.imshow("color", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    sock.sendto(msg.encode('utf-8'), (UDP_IP, UDP_PORT))
    print("Send message: " + msg)
    
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()