import numpy as np
import cv2
import sys
import socket
import time
sys.path.append('C:/Users/user/Downloads/opencv/sources/samples/python')

UDP_IP = "127.0.0.1"
UDP_PORT = 5065

print ("UDP target IP:", UDP_IP)
print ("UDP target port:", UDP_PORT)
sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) # UDP

font = cv2.FONT_HERSHEY_TRIPLEX

# red
lower_red = np.array([161, 100, 100])  
upper_red = np.array([179, 255, 255])
# orange
low_orange = np.array([10, 100, 100]) 
upper_orange = np.array([18, 255, 255])
# yellow
lower_yellow = np.array([26, 43, 46]) 
upper_yellow = np.array([33, 255, 255])   
# green has problem
lower_green = np.array([35, 43, 46])  
upper_green = np.array([77, 255, 255])  
# blue
lower_blue = np.array([93, 100, 100]) 
upper_blue = np.array([126, 255, 255])  
# brown

# gray
lower_gray = np.array([0,0,46])  
upper_gray = np.array([180, 43, 220])
# black
lower_black = np.array([0,0,0])  
upper_black = np.array([180, 255, 46])
# white
lower_white = np.array([0,0,221])
upper_white = np.array([180,30,255])

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
while True:
    ret, frame = cap.read()
    frame = cv2.flip( frame, 1 ) 
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv_img, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv_img,lower_yellow,upper_yellow)
    # mask_green = cv2.inRange(hsv_img, lower_green, upper_green) 
    # mask_gray = cv2.inRange(hsv_img, lower_gray, upper_gray)
    mask_blue = cv2.inRange(hsv_img, lower_blue, upper_blue)
    mask_black = cv2.inRange(hsv_img, lower_black, upper_black)
    mask_white = cv2.inRange(hsv_img, lower_white, upper_white)
# 篩選顏色
    mask_red = cv2.medianBlur(mask_red,7)   # 中值濾波
    mask_yellow = cv2.medianBlur(mask_yellow,7)
    # mask_green = cv2.medianBlur(mask_green, 7)  
    # mask_gray = cv2.medianBlur(mask_gray,7)
    mask_blue = cv2.medianBlur(mask_blue,7)
    mask_black = cv2.medianBlur(mask_black,7)
    mask_white = cv2.medianBlur(mask_white,7)
    contours_red, hierarchy_red = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, hierarchy_yellow = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours_green, hierarchy_green = cv2.findContours(mask_green, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # contours_gray, hierarchy_gray = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, hierarchy_blue = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_black, hierarchy_black = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_white, hierarchy_white = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    msg =""
# RED    
    flagR=0
    for red in contours_red:
        (x, y, w, h) = cv2.boundingRect(red)
        if w<30 and h<30:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Red", (x, y - 5), font, 0.7, (0, 0, 255), 2)
        flagR=1
    if(flagR):
        msg = msg+ "Red "
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
        msg = msg+ "Yellow "
# GREEN
    # flagG=0
    # for green in contours_green:
    #     (x, y, w, h) = cv2.boundingRect(green)
    #     if w<30 and h<30:
    #         continue
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv2.putText(frame, "Green", (x, y - 5), font, 0.7, (0, 255, 0), 2)
    #     flagG=1
    # if(flagG):
    #     msg = msg+ "Green "
# BLUE
    flagBU=0
    for blue in contours_blue:
        (x, y, w, h) = cv2.boundingRect(blue)
        if w<30 and h<30:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Blue", (x, y - 5), font, 0.7, (255, 0, 0), 2)
        flagBU=1
    if(flagBU):
        msg = msg+ "Blue "
# BROWN

# GRAY
    # flagGray=0
    # for gray in contours_gray:
    #     (x, y, w, h) = cv2.boundingRect(gray)
    #     if w<30 and h<30:
    #         continue
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), 2)
    #     cv2.putText(frame, "Gray", (x, y - 5), font, 0.7, (50, 50, 50), 2)
    #     flagGray=1
    # if(flagGray):
    #     msg = msg+ "Something Wrong... It's Gray. "    
# BLACK 
    flagBK=0
    for black in contours_black:
        (x, y, w, h) = cv2.boundingRect(black)
        if w<30 and h<30:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(frame, "Black", (x, y - 5), font, 0.7, (0, 0, 0), 2)
        flagBK=1
    if(flagBK):
        msg = msg+ "OH NO! Over Cooked! It's Black. "
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
        msg = msg+ "White "

    cv2.imshow("color", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    sock.sendto(msg.encode('utf-8'), (UDP_IP, UDP_PORT))
    print("Send message: " + msg)
    
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()