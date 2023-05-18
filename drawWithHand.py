import cv2
import time
import numpy as np
import handTrackingModule as htm
import math
import random
# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

def draw_erase_symbol(draw_mode):
    if draw_mode:
        image = cv2.imread(r"C:\Users\betha\PycharmProjects\handTrackingProject\pen_symbol.jpg")
    else:
        image = cv2.imread(r"C:\Users\betha\PycharmProjects\handTrackingProject\eraser_symbol.jpg")
    image = cv2.resize(image, (30, 30))
    x = wCam - image.shape[1]
    y = 0
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.moveWindow('Image', x, y)
    cv2.imshow('Image', image)


wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0 # previous time
noHandCount = 0
draw_mode = True # false for erase mode
pen_color = (255, 51, 51)
eraser_color = (0, 204, 255)

detector = htm.handDetector(detectionCon=0.7)
canvas = np.zeros((hCam, wCam, 3), dtype=np.uint8) # Create a blank canvas for drawing
prevFingerPos = None # store previous fingertip position
#draw_erase_symbol(draw_mode)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1) # mirror the image

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        x, y = lmList[8][1], lmList[8][2]  # Index finger

        if draw_mode:
            cv2.circle(img, (x, y), 10, pen_color, cv2.FILLED)  # Draw circle at fingertip (BGR)
            if prevFingerPos is not None:  # connect the line between this and previous position
                cv2.line(canvas, prevFingerPos, (x, y), pen_color, 3)
            prevFingerPos = (x, y)

            # when finger enters lower left corner, a random color will be generated
            if x > wCam - 50 and y > hCam - 50:
                pen_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        else: # eraser mode
            cv2.circle(img, (x, y), 10, eraser_color, cv2.FILLED)  # Draw circle at fingertip (BGR)
            radius = 20  # Define the radius of the erasing circle
            cv2.circle(canvas, (x, y), radius, (0, 0, 0), cv2.FILLED) # Erase the area around the fingertip using a circular mask

        # when putting two fingers together, draw-eraser mode switches
        x_thumb, y_thumb = lmList[4][1], lmList[4][2]  # thumb
        length = math.hypot(x - x_thumb, y - y_thumb)
        if length < 15:
            cx, cy = (x + x_thumb) // 2, (y + y_thumb) // 2  # center of the line
            draw_mode = not draw_mode
            #draw_erase_symbol(draw_mode)

    # if no hand detected after 50 frames, canvas will be cleared
    else:
        prevFingerPos = None
        noHandCount += 1
        if noHandCount == 100:
            canvas = np.zeros((hCam, wCam, 3), dtype=np.uint8)
            noHandCount = 0

    # Display the canvas and the image
    img = cv2.add(img, canvas)

    # Calculate the FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display the FPS on the image
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (32, 32, 32), 3)
    # Display draw/eraser mode
    if draw_mode:
        cv2.putText(img, "Draw mode", (wCam - 150, 70), cv2.FONT_HERSHEY_PLAIN, 2, pen_color, 3)
    else:
        cv2.putText(img, "Erase mode", (wCam - 150, 70), cv2.FONT_HERSHEY_PLAIN, 2, eraser_color, 3)
    cv2.rectangle(img, (wCam - 50, hCam - 50), (wCam - 1, hCam - 1), (255, 0, 0), 3)  # draw the frame of color changing space


    # Show the images
    cv2.imshow("Canvas", canvas)
    cv2.imshow("Image", img)

    # Check for key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()
