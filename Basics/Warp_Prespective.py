import cv2
import numpy as np

img = cv2.imread('Resources/cards.jpg')

width, height = 250,350
pts1 = np.float32([[111,219],[287,188],[154,482],[352,440]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgOutput = cv2.warpPerspective(img,matrix,(width,height))

for x in range (0,4):
    cv2.circle(img,(pts1[x][0],pts1[x][1]),15,(0,255,0),cv2.FILLED)

cv2.imshow("Original Image ", img)
cv2.imshow("Output Image ", imgOutput)
cv2.waitKey(0)
