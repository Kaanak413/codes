import cv2
import numpy as np 
from numpy import zeros


class calculateBonePixelNUmbers:
    def __init__(self,path,kernelSize,thresholdVal):
        self.path=path
        self.img=cv2.imread(self.path)
        self.grayscaledImg =cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        self.height,self.width = self.grayscaledImg.shape
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernelSize,kernelSize))
        self.thresholdVal=thresholdVal
        self.th,self.thresholdedImg = cv2.threshold(self.grayscaledImg,203,255,cv2.THRESH_BINARY)
        cv2.imshow("Img", self.grayscaledImg)
        cv2.imshow("ThresImg", self.thresholdedImg)
        cv2.waitKey(0) 

    def erosion(self,img):
        erosionImg = cv2.erode(img,self.kernel,iterations=1)
        cv2.imshow("ErosionImg", erosionImg)
        cv2.waitKey(0)
        return erosionImg 
    def getTheBonesPixelCount(self):
        res = self.erosion(self.thresholdedImg)
        
      
        # res = self.openThenClose()
        analysis = cv2.connectedComponentsWithStats(res, 
                                            4, 
                                            cv2.CV_32S) 
        (totalLabels, label_ids, values, centroid) = analysis 
        return totalLabels,values
    

bonePixel = calculateBonePixelNUmbers('images/Soru2.tif',5,127)    

totalLabels,values = bonePixel.getTheBonesPixelCount()
print(totalLabels,values)