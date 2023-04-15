import cv2
import mediapipe as np
import time

cap = cv2.VideoCapture(0)

npHands = np.solutions.hands
hands = npHands.Hands()
npDraw = np.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                # if id ==0:
                #     cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
                
                
            npDraw.draw_landmarks(img,handLms,npHands.HAND_CONNECTIONS)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(18,78),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)



    cv2.imshow("Image",img)
    cv2.waitKey(1)