##N Nearest Neighbour Test
import numpy as np
import math
import csv
import operator
import random
import time
from operator import itemgetter

def findNeighbors(trainX, testX, k):
	distanceList=[]
	D=17
	trainingSetSample=trainX[:, 1:D].astype(np.int)
	testSetPoint=testX[:, 1:D].astype(np.int)
	neighborList=[]
	for testS in range(len(testSetPoint)):
		distanceList=[]
		for x in range(len(trainingSetSample)):
			trainingTestData=trainingSetSample[x, :]
			dist=getDistanceOfPoints(trainingTestData, testSetPoint[testS])
			distanceList.append((trainX[x][0], dist))
		distanceList.sort(key=operator.itemgetter(1))
		#Tie the testData with the point
		neighborList.append((testX[testS][0], distanceList))
		distanceList.sort(key=operator.itemgetter(1))
	return neighborList


def getDistanceOfPoints(trainingTestData, testSetPoint):
	calDistance=0
	for i in range(0,16):
		calDistance=calDistance+pow((trainingTestData[i]-testSetPoint[i]), 2)
	sumDistance=math.sqrt(calDistance)
	return sumDistance

def testknn(trainX , trainY, testX, k):
	#find the nearest neighbour for all the training sets
	neighbors=findNeighbors(trainX, testX, 1)
	return neighbors

def condensedata(trainX, trainY):
	condensedSet=[]
	sampleCondensedSet=random.sample(trainingX, 1)
	neighbors=findNeighbors(sampleCondensedSet, testX, 1)
	accuracy=calculateAccuracy(neighbors, trainX)
	if not int(accuracy):
		newCondensedData=random.sample(trainingX, 1)
		condensedSet.append(newCondensedData)
	calculateAccuracy(condensedSet, testX, 1)	


def calculateAccuracy(fullList, nSample, k):
	success_count = 0
	total = len(fullList)
	orig = k
	for lst in fullList:
		check_flag = 1
		expected_result = lst[0]
		k = orig
		lst = lst[1][: k]
		point_list = list(map(itemgetter( 0 ), lst))
		val_count = point_list.count(expected_result)
		if val_count >= (k/2)+1:
			check_flag = 0
			success_count += 1
			continue
		processed_list = [expected_result]
		for point in point_list:
			if point in processed_list:
				continue
			nw_count = point_list.count(point)
			k -= nw_count
			if nw_count > val_count:
				check_flag = 0
				break
			if val_count > math.ceil(float(k/2.0)):
				check_flag = 0
				success_count += 1
				break
			processed_list.append(point)
		if check_flag and point_list[0] == expected_result:
			success_count += 1
		continue
	predictionAccuracy=float(success_count)/total
	print('Prediction  Percentage for K= {0} and Sample Size {1} is {2}'.format(orig,nSample,predictionAccuracy*100))
	return predictionAccuracy

def main():
	testSet=[]
	trainingSet=[]
	#with open('C:\Users\Gaurav\Desktop\letter-recognition.data.txt', 'rb') as f:
	with open('letter-recognition.data.txt', 'rb') as f:
		reader = csv.reader(f)
		i=0
		for row in reader:
			if (i < 15000):
				trainingSet.append(row)
			elif (i>=15000):
				testSet.append(row)
			i=i+1
	D=17
	kList=[1,3,5,7,9]
	
	N=[100,1000,2000,5000,10000,15000]
	
	##Form the testing Data matrix from testSet
	nTest=testSet
	nTestNumpyArray=np.array(nTest)
	testX=nTestNumpyArray[:, 0:D]
	testY=nTestNumpyArray[:, 0:1]

	#Form the training Matrix from trainingData
	for nSample in N:
		nTrain=random.sample(trainingSet, nSample)
		nTrainNumpyArray=np.array(nTrain)
		trainX=nTrainNumpyArray[:, 0:D]
		trainY=nTrainNumpyArray[:, 0:1]
		for k in kList:
			testY=testknn(trainX, trainY, testX, k)
			calculateAccuracy(testY,nSample, k)

		##The condensed algorithm is currently not working as expected and hence commented
		#However the function implementation is above

		#condensedata(trainX, trainY)
start_time = time.time()
main()
print("Time taken %s seconds" % (time.time() - start_time))
