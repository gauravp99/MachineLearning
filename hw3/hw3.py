##Author: Gaurav Pant
##Decision stump and PLA implementation

import numpy as np
import random
import csv
import math
from math import exp

def generateData(filename):
    data=[]
    with open(filename, 'rb') as f:
        reader=csv.reader(f)
        for row in reader:
            data.append(row)
    data=np.array(data)
    np.random.shuffle(data)
    size_of_training=int(0.75*len(data))
    random_training_data=data[range(0,size_of_training),:]
    random_test_data=data[range(size_of_training,len(data)),:]

    XTrain_pre,YTrain_pre=random_training_data[:,range(1,17)],random_training_data[:,0]
    #Generate XTrain, YTrain which is  a 75% of the training data given
    #All republican are given a positive label.Also for the missing value in the Xi
    # we consider it as the negative example and initialized it as -1 XTrain
    XTrain,YTrain=np.where(XTrain_pre=='y',1,-1), np.where(YTrain_pre=='republican',1,-1)

    XTest_pre,YTest_pre=random_test_data[:,range(1,17)],random_test_data[:,0]
    #Generate XTest, YTestActual which is  a 75% of the training data given
    XTest,YTestActual=np.where(XTest_pre=='y',1,-1), np.where(YTest_pre=='republican',1,-1)

    return XTrain,YTrain,XTest,YTestActual

def adaTrain(XTrain, YTrain, version):
   #returns a trained model as per the version of the algorithm
    if version=='stump':
        hypothesis_function=lambda x: 1 if (x[0]==1) else -1
        N=len(XTrain)
        Di=np.ones(N) / N
        alpha_list=[]
        D=16
        weak_classifier_list=[]
        hypothesis_function_list=[]
        weak_learner_list=[]

        for d in range(D):
            errors=[]
            weak_classifier_index = getweaklearner(XTrain,YTrain,weak_learner_list)
            if weak_classifier_index:
                hypothesis_function= lambda x, y: 1 if (x[y]==1) else -1
                hypothesis_function_list.append((hypothesis_function,weak_classifier_index))
                weak_classifier_list.append(weak_classifier_index)
                for i in range(len(XTrain)):
                    if YTrain[i]==hypothesis_function(XTrain[i],weak_classifier_index):
                        errors.append(0)
                    else:
                        errors.append(1)
                np_errors=np.array(errors)
                e=(np_errors*Di).sum()
                alpha = 0.5 * np.log((1-e)/e)
                alpha_list.append(alpha)
                for i in range(len(XTrain)):
                    #Update weights
                    Di[i]=Di[i]*exp(-alpha*YTrain[i]*hypothesis_function(XTrain[i],weak_classifier_index))
                Di=Di/Di.sum()
        return alpha_list,hypothesis_function_list,version
    else:
        model=perceptron(XTrain, YTrain)
        return model

#Find a weak learner from the set of given attribute
# The error is in the limit of 0.5
def getweaklearner(XTrain, YTrain, weak_learner_list):
    weak_classifier_dimension=random.choice(list(enumerate(XTrain[0])))[0]
    if weak_classifier_dimension in weak_learner_list:
        return None
    else:
        weak_learner_list.append(weak_classifier_dimension)
    accuracy_count=0
    hypothesis_function= lambda x, y: 1 if (x[y]==1) else -1
    for i in range(len(XTrain)):
        if YTrain[i]==hypothesis_function(XTrain[i],weak_classifier_dimension):
            accuracy_count=accuracy_count+1
    error=float(len(XTrain)-accuracy_count)/len(XTrain)
    if error < 0.5:
        return weak_classifier_dimension
    else:
        getweaklearner(XTrain, YTrain, weak_learner_list)

#This will return the weight vector, iterations, version of the perceptron algorithm
# (if it is possible to converge) the data.
#This will also break after 1000 iterations
def perceptron(XTrain, YTrain):
    iteration=0
    max_iteration=0
    version='perceptron'
    w=np.zeros(17)
    XTrain=np.insert(XTrain,0,1,axis=1)
    w_list=w
    min_error=1.0
    while (misclassified(w,XTrain,YTrain) ):
        if max_iteration > 1000:
            return w_list,iteration,version
        misclassified_point=random.choice(getMisclassifiedPoint(w,XTrain,YTrain))
        x_new=misclassified_point[0]
        y_new=misclassified_point[1]
        w=w+y_new*x_new
        iteration=iteration + 1
        error=calculate_error_pocket(XTrain, YTrain, w)
        if error < min_error:
            min_error=error
            w_list=w
        max_iteration=max_iteration+1
    return w_list,iteration,version

#Calcualte the error percentage of the pocket perceptron
#algorithm. This is needed to store the best possible weights with
#minimum error
def calculate_error_pocket(XTrain, YTrain, w):
    mis_classified_count=0
    for i in range(len(XTrain)):
        point_sign=w.T.dot(XTrain[i])
        if (cmp(point_sign,0) != YTrain[i]):
            ##Means the w is still not correct and if we encounter a misclassfied point with this then break and exit
            mis_classified_count=mis_classified_count+1
    return float(mis_classified_count)/len(XTrain)

#This will return the list of all the misclassified point with labels(if any) from the sample
def getMisclassifiedPoint(w,XTrain,YTrain):
    mis_point=[]
    for i in range(len(XTrain)):
        point_sign=w.T.dot(XTrain[i])
        if (cmp(point_sign,0) != YTrain[i]): #Cmp retrurns the -1,0,+1 for positive,zero and negative labels
            mis_point.append((XTrain[i],YTrain[i]))
    return (mis_point)

#If there exists a misclassified point left then this method will return true otherwise false.
#The weighted vector is the updated weight after a misclassified point is identified.
def misclassified(w,X,Y):
    flag=False  #flag to test if there are still misclassified points left in the sample
    for i in range(len(X)):
        point_sign=w.T.dot(X[i])
        if (cmp(point_sign,0) != Y[i]):
            ##Means the w is still not correct and if we encounter a misclassfied point with this then break and exit
            flag=True
            break
    return flag


def adaPredict(model, XTest):
    YTest = []
    w,iterations,version=model
    if version=='perceptron':

        XTest=np.insert(XTest,0,1,axis=1)
        for i in range(len(XTest)):
            point_sign=w.T.dot(XTest[i])
            if cmp(point_sign, 0) == 1:
                YTest.append(1)
            else:
                YTest.append(-1)
    else:
        alpha_list=w
        hf=iterations
        for t in range(len(XTest)):
            #for i in range(len(alpha_list)):
            #final_hypothesis=alpha_list[i]*hypothesis_function_list[i][0](XTest[t],hypothesis_function_list[i][1]) for i in range(len(alpha_list))])
            final_hypothesis=[(alpha_list[i])*(hf[i][0](XTest[t],hf[i][1])) for i in range(len(alpha_list))]
            if cmp(sum(final_hypothesis), 0) == 1:
                YTest.append(1)
            else:
                YTest.append(-1)
    return YTest

#Predict the accuracy of the model with the actual and predicted labels
def calculate_model_accuracy(YTest,YTestActual):
    YLabelAccuracy=YTest == YTestActual
    accurate_label=np.count_nonzero(YLabelAccuracy)
    accuracy=float(accurate_label)/len(YTest)*100
    return accuracy


def main():
    ##Please change the file name accordingly
    file_name='house-votes-84.data.txt'
    XTrain, YTrain, XTest, YTestActual =generateData(file_name)
    #version='stump'
    
    version='perceptron'
    model = adaTrain(XTrain, YTrain, version)
    YTest = adaPredict(model, XTest)
    accuracy=calculate_model_accuracy(YTest, YTestActual)
    print "Version %s has %f accuracy " %(version,accuracy)

main()