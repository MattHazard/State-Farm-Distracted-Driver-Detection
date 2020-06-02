from PIL import Image
import os

nTrain = r"C:\Users\Matt\Desktop\imgs\sTrain"
nTest = r"C:\Users\Matt\Desktop\imgs\sTest"
trainFolder = r"C:\Users\Matt\Desktop\imgs\train"
testFolder = r"C:\Users\Matt\Desktop\imgs\test"
nSize = (80, 60)

#convert training images
for folder in os.listdir(trainFolder):
    for filename in os.listdir(trainFolder+"\\"+folder):
        oLoc = trainFolder+"\\"+folder+"\\"+filename
        nLoc = nTrain+"\\"+folder+"\\"+filename
        img = Image.open(oLoc)
        nImg = img.resize(nSize)
        nImg.save(nLoc)
        img.close()
        nImg.close()
#convert test images
for filename in os.listdir(testFolder):
    oLoc = testFolder+"\\"+filename
    nLoc = nTest+"\\"+filename
    img = Image.open(oLoc)
    nImg = img.resize(nSize)
    nImg.save(nLoc)
    img.close()
    nImg.close()