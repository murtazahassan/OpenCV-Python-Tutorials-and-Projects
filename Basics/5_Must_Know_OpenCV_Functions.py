
import cv2
import numpy as np

kernel = np.ones((5,5),np.uint8)
print(kernel)

path = "Resources/lena.png"
img =  cv2.imread(path)
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),0)
imgCanny = cv2.Canny(imgBlur,100,200)
imgDilation = cv2.dilate(imgCanny,kernel , iterations = 10)
imgEroded = cv2.erode(imgDilation,kernel,iterations=2)

cv2.imshow("Lena",img)
cv2.imshow("GrayScale",imgGray)
cv2.imshow("Img Blur",imgBlur)
cv2.imshow("Img Canny",imgCanny)
cv2.imshow("Img Dialation",imgDilation)
cv2.imshow("Img Erosion",imgEroded)
cv2.waitKey(0)
