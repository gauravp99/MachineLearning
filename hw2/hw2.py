##Author: Gaurav Pant
##PLA with and without linear regression

import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.linalg import inv


##Method Name: pla
#This will return the weight vector and the iterations taken to converge the data.
#Input: X, Y, w0
#Output: w,iteration

def pla(X, Y, w0):
    iteration=0
    w=w0
    X=np.insert(X,0,1,axis=1)
    while (misclassified(w,X,Y)):
        misclassified_point=random.choice(getMisclassifiedPoint(w,X,Y))
        x_new=misclassified_point[0]
        y_new=misclassified_point[1]
        w=w+y_new*x_new
        iteration=iteration + 1
    return w,iteration


##Method Name: getMisclassifiedPoint
#This will return the list of all the misclassified point with labels(if any) from the sample
#Input: w,X,Y
#Output: List of misclasified point with labels

def getMisclassifiedPoint(w,X,Y):
    mis_point=[]
    for i in range(len(X)):
        point_sign=w.T.dot(X[i])
        if (cmp(point_sign,0) != Y[i]): #Cmp retrurns the -1,0,+1 for positive,zero and negative labels
            mis_point.append((X[i],Y[i]))
    return (mis_point)


##Method Name: misclassified
#This method will return a boolean value.
#Input: w, X, Y
#Output: True or False
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


##Method Name: psuedoinverse
#Calculate the psuedoinverse of X
#Input: X and Y
#Output: learnt weight vector w

def pseudoinverse(X,Y):
    X=np.insert(X,0,1,axis=1) #Aded the x0 to the input vector
    w = (np.linalg.pinv(X)).dot(Y)
    return w

#Method Name: generate Data
#Generate Data from (N,2) matrix
#Input : N
#Output: X and Y

def generateData(N):
    X=np.random.uniform(-1,1, (N,2))
    line=np.random.uniform(-1,1, (2,2))
    Y=[]
    for i in range(len(X)):
        point_location=((line[1][0]-line[0][0])*(X[i][1]-line[0][1]) - (line[1][1]-line[0][1])*(X[i][0]-line[0][1]))
        if point_location >= 0:
            Y.append(1)
        elif point_location < 0:
            Y.append(-1)
    return (X,Y)

#Method Name: plot_chart
#This generates a chart with given (X,Y)
#Points with label -1 are  marked green and remaining points are marked red.
#Input : X,Y
#output: Plot the points in the chart

def plot_chart(X,Y):
    for i in range(len(X)):
        if Y[i]==1:
            plt.plot(X[i][0],X[i][1],'go')
        else:
            plt.plot(X[i][0],X[i][1],'ro')

def main():
    N=20
    X,Y = generateData(N)
    w0=np.zeros(3)
    mean_pla_sum=0
    mean_pla_sum_regressions=0
    for i in range(100):
        X,Y = generateData(N)
        w,iters=pla(X, Y, w0)
        mean_pla_sum=mean_pla_sum+iters

        ##Psuedo Inverse Implementation to learn weights
        w1 = pseudoinverse(X, Y)
        w2,iters=pla(X, Y, w1)
        mean_pla_sum_regressions=mean_pla_sum_regressions+iters

    #print "w=%s,Iterations=%d,N=%d for PLA without linear regressions" %(w,mean_pla_sum/100,N)
    #print "w=%s,Iterations=%d N=%d for PLA with psuedo inverse" %(w2,mean_pla_sum_regressions/100,N)

    plot_chart(X,Y)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.plot([-w[0]/w[1],0],[0,-w[0]/w[2]],linestyle='dashed', color='blue')
    plt.plot([-w2[0]/w2[1],0],[0,-w2[0]/w2[2]],linestyle='dashed', color='green')

    #The below piece of code is intentionally commented to avoid displaying the charts.
    #plt.show()

main()
