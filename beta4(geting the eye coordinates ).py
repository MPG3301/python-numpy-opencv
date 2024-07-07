from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import dlib
import cv2
import pyautogui as m
def calculateView(x,y):
    xsMax, ysMax = m.size()                          #Screen maximum size fetching location
    xsMin, ysMin = 0, 0                              #Screen origin Location
    xfMax, xfMin = 370, 270                          #Frame x width
    yfMax, yfMin = 290, 200                          #Frame y width
    sx = (xsMax - xsMin) // (xfMax - xfMin)          #Adjusting of the frame x width accordingly with the screen
    sy = (ysMax - ysMin) // (yfMax - yfMin)          #Adjusting of the frame y width accordingly with the screen
    xv = xsMin + (x - xfMin) * sx                    #calculating of the x coordinates accordingly with the frame and screen
    yv = ysMin + (y - yfMin) * sy                    #calculating of the y coordinates accordingly with the frame and screen
    return xv,yv   
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"   
print("[INFO] loading facial landmark predictor...") 
detector = dlib.get_frontal_face_detector()                   #initializing dlib face detector
predictor = dlib.shape_predictor(PREDICTOR_PATH)  
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] #getting the indexes of the facial landmarks for the left eye
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]#getting the indexes of the facial landmarks for the right eye
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]     #getting the indexes of the facial landmarks for the nose11:47 22-05-2021
vs = cv2.VideoCapture(0)  
while True:
    
    ret, frame = vs.read()                                    #read a video frame by frame, read() returns tuple in which 1st item is boolean value either True or False and 2nd item is frame of the video
                                                              # read() returns False when live video is ended so no frame is readed and error will be generated
    frame = cv2.flip(frame, 1)                                #to flip the frame horizontally for perfect movement of the mouse directiion wise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)            # detect face in the grayscale frame
    rects = detector(gray, 0)                                 #dlib face detector detects the face in gray scale
    
    for rect in rects: 
        shape = predictor(gray, rect)                         #determine the facial landmarks for the face region then convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = face_utils.shape_to_np(shape)                 #then convert the facial landmark (x, y)-coordinates to a NumPy array
        leftEye = shape[lStart:lEnd]                          #getting the left eye coordinates array
        nose = shape[nStart:nEnd]                             #getting the nose coordinates array
        rightEye = shape[rStart:rEnd]                         #getting the right eye coordinates array
        
        l1 = len(leftEye)
        l2 = len(rightEye)
        l3 = len(nose)
        xl, yl = leftEye[0]
        xr, yr = rightEye[0]
        xn, yn = nose[0]
        xl1,yl1 = leftEye[3]
        xr1,yr1 = rightEye[3]
        xn1,yn1 = nose[4]
        x = (xl + xr + xn + xl1 + xr1 + xn1)/6
        y = (yl + yr + yn + yl1 + yr1 + yn1)/6
        #getting eye coordinates
        x1 = np.int(x)
        y1 = np.int(y)
        x,y = calculateView(x1,y1)
        print('x=',x ,'y=',y)
       
 # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1000) & 0xFF
 
    # if the `q` or 'esc' key was pressed, break from the loop
    if key == ord("q"):
        break
    elif key == 27:
        break
 
cv2.destroyAllWindows()
 
