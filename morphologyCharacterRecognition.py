import cv2
import numpy as np 
from numpy import zeros


class MorphologyCharacterRecognition:
    def __init__(self,path,kernelSize,thresholdVal):
        self.path=path
        self.img=cv2.imread(self.path)
        self.grayscaledImg =cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        self.height,self.width = self.grayscaledImg.shape
        self.kernel = np.ones((kernelSize,kernelSize),np.uint8)
        self.thresholdVal=thresholdVal

    def opening(self,img):
        opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,self.kernel)
        return opening
    
    def closing(self,img):
        closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,self.kernel)
        return closing
    
    def openThenClose(self):
        res = self.closing(self.opening(self.grayscaledImg))
        cv2.imshow("Result", res)
        cv2.waitKey(0) 
        return res
    def getTheNumberOfCharacters(self):
        res = self.openThenClose()
        returnVal = cv2.connectedComponentsWithStats(res)
        return returnVal    
    


