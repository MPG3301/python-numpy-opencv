from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import dlib
import cv2
import pyautogui as m

m.FAILSAFE = False
(scrx,scry)=m.size()
mLocOld = np.array([0,0])
mouseLoc = np.array([0,0])
DampingFactor = 15          
def calculateView(x,y):
    xvMax, yvMax = m.size()
    xvMin, yvMin = 0, 0
    xwMax, xwMin = 370, 270
    ywMax, ywMin = 290, 200
    sx = (xvMax - 0) // (xwMax - xwMin)
    sy = (yvMax - 0) // (ywMax - ywMin)
    xv = xvMin + (x - xwMin) * sx
    yv = yvMin + (y - ywMin) * sy
    return xv,yv
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
EYE_AR_THRESH = .25
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
print("[INFO] starting video stream thread...")
vs = cv2.VideoCapture(0)
while True:
    ret, frame = vs.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        nose = shape[nStart:nEnd]
        rightEye = shape[rStart:rEnd]
        print('nose')
        print(nose[0])
        xv, yv = nose[0]
        xw = np.int(xv)
        yw = np.int(yv)
        print(type(xv))
        xv,yv = calculateView(xw,yw)       
        mouseLoc = mLocOld + ((xv,yv)-mLocOld)//DampingFactor
        print('nx = {} and ny = {}'.format(mouseLoc[0], mouseLoc[1]))
        m.moveTo(mouseLoc[0],mouseLoc[1])
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        #cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)
        print("right: ",rightEAR,"Left: ",leftEAR)
        if rightEAR < .26:
            m.click(mouseLoc[0],mouseLoc[1],clicks = 1, button = 'left' )
        if leftEAR < .26:
            m.click(mouseLoc[0],mouseLoc[1],clicks = 1, button = 'left')
        mLocOld = mouseLoc
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == 27:
        break
cv2.destroyAllWindows()
