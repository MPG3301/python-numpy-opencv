from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import dlib
import cv2
import pyautogui as m
import RPi.GPIO as GPIO
import time

sensor = 16
GPIO.setmode(GPIO.BOARD)
GPIO.setup(sensor,GPIO.IN)

m.FAILSAFE = False                                   #Statement to avoid quick movement of the mouse as far up and left as we can 
(scrx,scry)=m.size()                                 #our screen size ie either tv or monitor or laptop

mouseOldLocation = np.array([0,0])
mouseNewLocation = np.array([0,0])
DampingFactor = 15                                   #smoothness in the movement of mouse 

def calculateView(x,y):
    xsMax, ysMax = m.size()                          #Screen maximum size fetching location
    xsMin, ysMin = 0, 0                              #Screen origin Location
    xfMax, xfMin = 370, 270                          #Frame x width
    yfMax, yfMin = 290, 200                          #Frame y width
    sx = (xsMax - xsMin) // (xfMax - xfMin)          #Adjusting of the frame x width accordingly with the screen
    sy = (ysMax - ysMin) // (yfMax - yfMin)          #Adjusting of the frame y width accordingly with the screen
    xv = xsMin + (x - xfMin) * sx                    #Adjusting of the frame x coordinates accordingly with the screen
    yv = ysMin + (y - yfMin) * sy                    #Adjusting of the frame y coordinates accordingly with the screen
    return xv,yv                                     #Giving Location of the frame 

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])               #calculating the vetical distance of the eye 
    B = dist.euclidean(eye[2], eye[4])               #calculating the vetical distance of the eye 
    C = dist.euclidean(eye[0], eye[3])               #calculating the horizontal distance of the eye
    eyeAspectRatio = (A + B) / (2.0 * C)             #calculating the eye aspect ratio
    return eyeAspectRatio                            #Giving eye aspect ratio

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat" 
EYE_ASPECT_RATIO_THRESHOLD = 0.25                    #threshold value to identify the eye blink
EYE_ASPECT_RATIO_CONSEC_FRAMES = 3                   #number of consecutive frames the eye must be below the threshold
TOTAL1 = 0                                           #variable will be used for functions including two or more blinks
TOTAL2 = 0                                           #variable will be used for functions including two or more blinks

print("[INFO] loading facial landmark predictor...")

detector = dlib.get_frontal_face_detector()                   #initializing dlib face detector
predictor = dlib.shape_predictor(PREDICTOR_PATH)              #using shape predictor to create facial landmarks and identify the facial characteristics

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] #getting the indexes of the facial landmarks for the left eye
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]#getting the indexes of the facial landmarks for the right eye
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]     #getting the indexes of the facial landmarks for the nose
print("[INFO] starting video stream thread...")
vs = cv2.VideoCapture(0)    
               
try: 
   while True:
      if GPIO.input(sensor):
                                         #starting of the live streaming using webcam
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
            # l1 = len(leftEye)
            #print(l1)=6
            #l2 = len(rightEye)
            #print(l2)=6
            #l3 = len(nose)
            #print(l3)=9
            xl, yl = leftEye[0]
            xr, yr = rightEye[0]
            xn, yn = nose[0]
            xl1,yl1 = leftEye[3]
            xr1,yr1 = rightEye[3]
            xn1,yn1 = nose[4]
            x = (xl + xr + xn + xl1 + xr1 + xn1)/6
            y = (yl + yr + yn + yl1 + yr1 + yn1)/6
            #getting frame coordinates  
            x1 = np.int(x)
            y1 = np.int(y)
            x,y = calculateView(x1,y1) 
         
            # for mouse control              
            mouseNewLocation = mouseOldLocation + ((x,y)-mouseOldLocation)//DampingFactor
            print('nx = {} and ny = {}'.format(mouseNewLocation[0], mouseNewLocation[1]))
            m.moveTo(mouseNewLocation[0],mouseOldLocation[1])
    
            # calculating the eye-aspect-ratio(EAR) for both eyes
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
 
            # compute the convex hull for the left and right eye, then visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
      
            print("right: ",rightEAR,"Left: ",leftEAR)
            if rightEAR < .2:
                m.click(mouseNewLocation[0],mouseNewLocation[1],clicks = 2, button = 'left' )
            
            if leftEAR < .2:
                m.click(mouseNewLocation[0],mouseNewLocation[1],clicks = 2, button = 'left')
            
            if (rightEAR < .2 and not leftEAR < .2) or (leftEAR < .2 and not rightEAR < .2): #xor operation
                m.scroll(4) # scroll up
                print('scrolling down')
            elif rightEAR < .2 and  leftEAR < .2:
                m.scroll(-4) # scroll down
                print('scrolling up')
            mouseOldLocation = mouseNewLocation

            # show the frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(250) & 0xFF
 
            # if the `q` or 'esc' key was pressed, break from the loop
            if key == ord("q"):
               break
            elif key == 27:
               break
      else:
           print('no operation')
except KeyboardInterrupt:
       GPIO.cleanup()
 
cv2.destroyAllWindows()
