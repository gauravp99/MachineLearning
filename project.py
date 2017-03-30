##Author Gaurav Pant
##Hotel Image Classification

##Credits: Using skimage package for hog


import cv2
from skimage.feature import hog
from skimage import data, color, exposure
import os
import glob
import csv
import numpy as np
from sklearn import neighbors, datasets
from sklearn import svm
from os.path import splitext
import random


#Generate hog:
#Step 1 : Normalized Image
#Step 2: Computing Gradient Histogram
#Step 3: Normalizing across blocks
#Step 4: Flattening into a feature Vector
#Step 5: Return the list of feature Vector

def generate_hog(x):
    train_images=[]
    X_Train=[]
    os.chdir("C:/Users/Gaurav/Downloads/train/")
    for file in glob.glob("*.jpg"):
        train_images.append(file)
    for image in x:
        gray=cv2.imread(str(image)+'.jpg', 0)
        image = color.rgb2gray(gray)
        res=cv2.resize(image,(256,256), interpolation=cv2.INTER_CUBIC)
        fd, hog_image = hog(res, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualise=True)
        X_Train.append(fd)
    return X_Train




##GenerateData
##Out of the total sample images randomly
## 35k images are chosen for training the model
## and rest of the images are chosen for testing purpose
## This can be changed accordingly if neeeded.
# Also for performance we use a smaller subset of the images

def generateData():
    x=[]
    y=[]
    data=[]
    i=0
    os.chdir("C:/Users/Gaurav/Downloads/")
    ##Added the faulty image which are broken from the training set.
    faulty=[6510, 7225, 8964, 11402, 11983, 15623, 22810, 23210, 25215, 28094, 28945, 31392, 36911, 38004, 39816, 44140, 49405, 49750, 51105, 51194]
    with open('train.csv', 'rb') as f:
        reader=csv.reader(f)
        for row in reader:
            i=i+1
            if (i==1):
                continue
            else:
                dummy_list=map(int, row[1:9])
                y_label_1=[ix for ix, e in enumerate(dummy_list) if e != 0]
                if y_label_1:
                    if int(row[0]) in faulty:
                        continue
                    x.append(row[0])
                    y_label=y_label_1[0] + 1
                    y.append(y_label)
    combined = zip(x, y)
    random.shuffle(combined)
    #x[:], y[:] = zip(*combined)
    x_random=x[0:35000]
    y_random=y[0:35000]

    x_test_random=x[35000:len(x)]
    y_test_random=y[35000:len(y)]

    return x_random,y_random,x_test_random,y_test_random

##generatePredictionCSV
##This generates the prediction for the test data in CSV format
##The same can be submitted to Kaggle.

def generatePredictionCSV(x_test, y_test):
    with open("C:/Users/Gaurav/Downloads/submission.csv", "wb") as f:
        writer = csv.writer(f)
        header = ['id', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8']
        writer.writerow(header)

        for i in range(len(y_test)):
            names = [0]*9
            names[0] = x_test[i]
            names[1:9]=y_test[i][0]
            writer.writerow(names)

##This method generates the predicted label
##of the test images using the SVM classifier
##The output will be a list of the xtest, ytest
##where xtest is the image name and ytest is the
##probabilty of the 8 classes.

def predicttestdata(clf):
    test_images=[]
    y_test=[]
    x_test=[]
    os.chdir("C:/Users/Gaurav/Downloads/test/")
    for file1 in glob.glob("*.jpg"):
        test_images.append(file1)

    for test_image in test_images:

        gray = cv2.imread(test_image, 0)
        if gray is not None:

            image = color.rgb2gray(gray)
            res=cv2.resize(image,(256,256), interpolation=cv2.INTER_CUBIC)
            fd, hog_image = hog(res, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), visualise=True)
            y_testlabel=clf.predict_proba(fd.reshape(1, -1))
            y_test.append(y_testlabel)
            x_test.append(splitext(test_image)[0])
        else:
             print test_image
             x_test.append(splitext(test_image)[0])
             b=[0.125]*8
             y_test.append([b])
    return x_test,y_test


def main():

    x,y,x_test_random,y_test_random=generateData()
    print "----GenerateData done"
    X_Train=generate_hog(x)

    #X_Test will be used for cross validation
    X_Test=generate_hog(x_test_random)

    #Here the value of C is modified to get different accuracy of the model
    #Value of C changes the accuracy of the model . Higher C value means
    # a more complex model. We changed the value from C=1 to C=1000

    C=1.0

    #SVM is used as a classifier here with varying C values
    #C define the complexity of the model

    clf = svm.SVC(kernel='linear', C=C, probability=True)
    clf.fit(X_Train, y)
    Z=clf.predict_proba(X_Train)
    accuracy=clf.score(X_Test,y_test_random)

    print "----Prediction done"
    print ("Predicted model accuracy with cross validation set: "+ str(accuracy))

    #Use the classifer to generate the predicted label for
    #the testing data.

    xtest,ytest=predicttestdata(clf)
    print "----Prediction of Test data done"

    #Generate a CSV for the train data
    generatePredictionCSV(xtest, ytest)
    print "----Generation of CSV done"

main()