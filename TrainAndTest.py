# TrainAndTest.py

import cv2
import numpy as np
import operator
import os
from numpy import genfromtxt

import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import pandas as pd
from pandas import DataFrame


# module level variables ##########################################################################
#MIN_CONTOUR_AREA = 100
MIN_CONTOUR_AREA = 40

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
class ContourWithData():

    # member variables ############################################################################
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        return True

###################################################################################################
def main():
    allContoursWithData = []                # declare empty lists,
    validContoursWithData = []              # we will fill these shortly
    imgcount = 1
    numtemp = []

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # read in training classifications
    except:
        print("error, unable to open classifications.txt, exiting program")
        os.system("pause")
        return
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # read in training images
    except:
        print("error, unable to open flattened_images.txt, exiting program")        
        os.system("pause")
        return
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train
    #print(npaClassifications.shape())
    print(npaClassifications)

    kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    imgTestingNumbers = cv2.imread("test1.png")          # read in testing numbers image

    if imgTestingNumbers is None:                           # if image was not read successfully
        print("error: image not read from file")        # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit function (which exits program)
    # end if

    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)       # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                    # blur

                                                        # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)                                    # constant subtracted from the mean or weighted mean

    imgThreshCopy = imgThresh.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image

    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points

    for npaContour in npaContours:                             # for each contour
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data
    # end for

    for contourWithData in allContoursWithData:                 # for all contours
        if contourWithData.checkIfContourIsValid():             # check if valid
            validContoursWithData.append(contourWithData)       # if so, append to valid contour list
        # end if
    # end for

    #validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right
    validContoursWithData.sort(key = operator.attrgetter("intRectY"))         # sort contours from left to right

    strFinalString = ""         # declare final string, this will have the final number sequence by the end of the program

    for contourWithData in validContoursWithData:            # for each contour
                                                # draw a green rect around the current char
        cv2.rectangle(imgTestingNumbers,                                        # draw rectangle on original testing image
                      (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                      (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                      (0, 255, 0),              # green
                      2)                        # thickness

        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]
        
        cv2.imwrite("img//" + str(imgcount) + " x= " + str(contourWithData.intRectX) + "  y= " + str(contourWithData.intRectY) + " h= " + str(contourWithData.intRectHeight) + " w= " + str(contourWithData.intRectWidth) +".jpg" ,imgROI)
        #cv2.imwrite("img//imgtesting.jpg" ,imgROI)


        imgcount = imgcount + 1
        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage

        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

        npaROIResized = np.float16(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats
        #print(str(imgcount -1) + str(npaROIResized))
        if (imgcount <= 2):
            numtemp = np.array(npaROIResized)
        else:
            numtemp = np.concatenate((numtemp,npaROIResized))

        #retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)     # call KNN function find_nearest

        #strCurrentChar = str(chr(int(npaResults[0][0])))                                             # get character from results

        #strFinalString = strFinalString + strCurrentChar            # append current char to full string
    # end for

    print(strFinalString)                  # show the full string
    
    #numtemp = numtemp /255
    #numtemp = np.around(numtemp, decimals=2)
    c = np.savetxt('tempinput.csv', numtemp, delimiter =',')

    cv2.imshow("imgTestingNumbers", imgTestingNumbers)      # show input image with green boxes drawn around found digits
    cv2.imwrite("testing1.jpg",imgTestingNumbers)
    cv2.waitKey(0)                                          # wait for user key press

    cv2.destroyAllWindows()             # remove windows from memory

    return

###################################################################################################
def readcsv(rownum):
    num_data = genfromtxt('tempinput.csv', delimiter=',')
    imgROIResized= num_data[rownum]
    imgROIResized = imgROIResized.reshape(30,20)

    #imgROIResized = cv2.resize(imgROIResized, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage
    print(imgROIResized)
    cv2.imshow("imgTestingNumbers", imgROIResized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def kerasmodel():
    # The competition datafiles are in the directory ../input
    # Read competition data files:
    # Backend - Theano
    train = pd.read_csv("tempinput.csv")
    test  = pd.read_csv("test.csv")

    # Write to the log:
    print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
    print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
    # Any files you write to the current directory get shown as outputs

    np.random.seed(1337) 

    batch_size = 2
    nb_classes = 6
    nb_epoch = 20

    # convert class vectors to binary class matrices
    X_train=train.iloc[:,1:].as_matrix().astype('float32')
    X_test=test.as_matrix().astype('float32')
    X_train /= 255
    X_test /= 255
    #y_train2 = [1, 0, 3, 4, 5, 0, 2, 1]
    y_train2 = train.iloc[:,0:1]
    print(train.iloc[:,0:1])
    Y_train = np_utils.to_categorical(y_train2, nb_classes)
    
    print(Y_train)

    model = Sequential()
    model.add(Dense(512, input_shape=(600,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer= RMSprop(),
                metrics=['accuracy'])

    fit1 = model.fit(X_train, Y_train,batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)
    pred=model.predict(X_test)
    pred2=[]
    for i in range(pred.shape[0]):
        pred2.append(list(pred[i]).index(max(pred[i])))
    out_file = open("predictions.csv", "w")
    out_file.write("ImageId,Label\n")
    for i in range(len(pred2)):
        out_file.write(str(i+1) + "," + str(int(pred2[i])) + "\n")
    out_file.close()

if __name__ == "__main__":
    #readcsv(3)
    kerasmodel()
    #main()
# end if













# # TrainAndTest.py

# import cv2
# import numpy as np
# import operator
# import os

# # module level variables ##########################################################################
# MIN_CONTOUR_AREA = 100

# RESIZED_IMAGE_WIDTH = 20
# RESIZED_IMAGE_HEIGHT = 30

# ###################################################################################################
# class ContourWithData():

#     # member variables ############################################################################
#     npaContour = None           # contour
#     boundingRect = None         # bounding rect for contour
#     intRectX = 0                # bounding rect top left corner x location
#     intRectY = 0                # bounding rect top left corner y location
#     intRectWidth = 0            # bounding rect width
#     intRectHeight = 0           # bounding rect height
#     fltArea = 0.0               # area of contour

#     def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
#         [intX, intY, intWidth, intHeight] = self.boundingRect
#         self.intRectX = intX
#         self.intRectY = intY
#         self.intRectWidth = intWidth
#         self.intRectHeight = intHeight

#     def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
#         if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
#         return True

# ###################################################################################################
# def main():
#     allContoursWithData = []                # declare empty lists,
#     validContoursWithData = []              # we will fill these shortly

#     try:
#         npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # read in training classifications
#     except:
#         print("error, unable to open classifications.txt, exiting program")
#         os.system("pause")
#         return
#     # end try

#     try:
#         npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # read in training images
#     except:
#         print("error, unable to open flattened_images.txt, exiting program")
#         os.system("pause")
#         return
#     # end try

#     npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train

#     kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object

#     kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

#     imgTestingNumbers = cv2.imread("test1.png")          # read in testing numbers image

#     if imgTestingNumbers is None:                           # if image was not read successfully
#         print("error: image not read from file")        # print error message to std out
#         os.system("pause")                                  # pause so user can see error message
#         return                                              # and exit function (which exits program)
#     # end if

#     imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)       # get grayscale image
#     imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                    # blur

#                                                         # filter image from grayscale to black and white
#     imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
#                                       255,                                  # make pixels that pass the threshold full white
#                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
#                                       cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
#                                       11,                                   # size of a pixel neighborhood used to calculate threshold value
#                                       2)                                    # constant subtracted from the mean or weighted mean

#     imgThreshCopy = imgThresh.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image

#     imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
#                                                  cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
#                                                  cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points

#     for npaContour in npaContours:                             # for each contour
#         contourWithData = ContourWithData()                                             # instantiate a contour with data object
#         contourWithData.npaContour = npaContour                                         # assign contour to contour with data
#         contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
#         contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
#         contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
#         allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data
#     # end for

#     for contourWithData in allContoursWithData:                 # for all contours
#         if contourWithData.checkIfContourIsValid():             # check if valid
#             validContoursWithData.append(contourWithData)       # if so, append to valid contour list
#         # end if
#     # end for

#     validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right

#     strFinalString = ""         # declare final string, this will have the final number sequence by the end of the program

#     for contourWithData in validContoursWithData:            # for each contour
#                                                 # draw a green rect around the current char
#         cv2.rectangle(imgTestingNumbers,                                        # draw rectangle on original testing image
#                       (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
#                       (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
#                       (0, 255, 0),              # green
#                       2)                        # thickness

#         imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
#                            contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

#         imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage

#         npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

#         npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats

#         retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)     # call KNN function find_nearest

#         strCurrentChar = str(chr(int(npaResults[0][0])))                                             # get character from results

#         strFinalString = strFinalString + strCurrentChar            # append current char to full string
#     # end for

#     print(strFinalString)                  # show the full string

#     cv2.imshow("imgTestingNumbers", imgTestingNumbers)      # show input image with green boxes drawn around found digits
#     cv2.waitKey(0)                                          # wait for user key press

#     cv2.destroyAllWindows()             # remove windows from memory

#     return

# ###################################################################################################
# if __name__ == "__main__":
#     main()
# # end if









