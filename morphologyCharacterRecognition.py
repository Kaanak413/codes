import cv2
import numpy as np 
from numpy import zeros


class MorphologyCharacterRecognition:
    def __init__(self,path,kernelSize,thresholdVal):
        self.path=path
        self.img=cv2.imread(self.path)
        self.grayscaledImg =cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        self.height,self.width = self.grayscaledImg.shape
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
        self.thresholdVal=thresholdVal
        cv2.imshow("Img", self.grayscaledImg)
        cv2.waitKey(0) 
    def erosion(self,img):
        erosionImg = cv2.erode(img,self.kernel,iterations=1)
        cv2.imshow("ErosionImg", erosionImg)
        cv2.waitKey(0)
        return erosionImg 
    def dilation(self,img):
        dilatedImg = cv2.dilate(img,self.kernel,iterations=1)
        cv2.imshow("DilatedImg", dilatedImg)
        cv2.waitKey(0)
        return dilatedImg 
    def opening(self,img):
        opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,self.kernel)
        cv2.imshow("Open", opening)
        cv2.waitKey(0) 
        return opening
    
    def closing(self,img):
        closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,self.kernel)
        cv2.imshow("Close", closing)
        cv2.waitKey(0) 
        return closing
    
    def openThenClose(self):
        res = self.closing(self.opening(self.grayscaledImg))
        cv2.imshow("Result", res)
        cv2.waitKey(0) 
        return res
    def getTheNumberOfCharacters(self):
        self.erosion(self.grayscaledImg)
        self.dilation(self.grayscaledImg)
        res=self.closing(self.grayscaledImg)
      
        # res = self.openThenClose()
        analysis = cv2.connectedComponentsWithStats(res, 
                                            4, 
                                            cv2.CV_32S) 
        (totalLabels, label_ids, values, centroid) = analysis 
        return totalLabels    
    

ocr = MorphologyCharacterRecognition('images/Soru1.tif',2,127)

val = ocr.getTheNumberOfCharacters()
print( val)