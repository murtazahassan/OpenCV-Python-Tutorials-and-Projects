import numpy as np
from PIL import ImageGrab
import cv2


def captureScreen(bbox=(50,50,690,530)):
    capScr = np.array(ImageGrab.grab(bbox))
    capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
    return capScr


while True:
    timer = cv2.getTickCount()
    img = captureScreen()
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    cv2.putText(img,'FPS {}'.format(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 230, 20), 2)
    cv2.imshow('Screen Capture',img)
    cv2.waitKey(1)
