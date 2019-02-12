#coding:utf-8
from numpy import *

#载入数据
def loadDataSet():
	dataMat=[];labelMat=[]
	fr=open('testSet.txt')
	for line in fr.readlines():
		lineArr=line.strip().split()
		dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat,labelMat
def sigmoid(inX):
	return 1.0/(1+exp(-inX))
#梯度上升，传入特征矩阵dataMatIn和对应的标签classLabels,和循环次数
def gradAscent(dataMatIn,classLabels,maxCycles=500):
	dataMatrix=mat(dataMatIn)
	labelMat=mat(classLabels).transpose()
	m,n=shape(dataMatrix)
	alpha=0.001
	weights=ones((n,1))
	for k in range(maxCycles):
		h=sigmoid(dataMatrix*weights)#h=f(z)=1/(1+e(-z))-->logistic函数  z=w0*x0+w1*x1...
		error=(labelMat-h)
		weights=weights+alpha*dataMatrix.transpose()*error#用的极大似然法加梯度上升，并不是梯度上升
	return weights
#随机梯度上升，传入特征矩阵dataMatrix，标签classLabels和循环迭代次数numIter=150
def stocGradAscent0(dataMatrix,classLabels,numIter=150):
	dataMatrix=array(dataMatrix)
	m,n=shape(dataMatrix)
	#alpha=0.01
	weights=ones(n)
	for j in range(numIter):
		dataIndex = range(m)
		for i in range(m):
			alpha=4/(1.0+j+i)+0.01
			randIndex=int(random.uniform(0,len(dataIndex)))
			h=sigmoid(sum(dataMatrix[randIndex]*weights))
			error=classLabels[randIndex]-h
			#print dataMatrix[i]
			#print weights
			weights=weights+alpha*error*dataMatrix[randIndex]
			del(dataIndex[randIndex])

	return weights

#根据权重weights,特征矩阵dataMat，和标签labelMat画图
def plotBestFile(weights,dataMat,labelMat):
	import matplotlib.pyplot as plt
	weights=array(weights)
	#dataMat,labelMat=loadDataSet()
	dataArr=array(dataMat)
	n=shape(dataArr)[0]
	xcord1=[];ycord1=[]
	xcord2=[];ycord2=[]
	for i in range(n):
		if int(labelMat[i])==1:
			xcord1.append(dataArr[i,1])
			ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1])
			ycord2.append(dataArr[i,2])
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
	ax.scatter(xcord2,ycord2,s=30,c='green')
	x=arange(-3.0,3.0,0.1)
	#print weights[0]
	y=(-weights[0]-weights[1]*x)/weights[2]
	#y1=y.reshape((60))
	#print y[0]==y
	#print y,shape

	ax.plot(x,y)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()
#********************上面为logistics算法***********下面是根据logistics得到权重W，再根据logistics判断测试数据的所属类别

#传入测试数据经过logistics公式后的得分和权重得到所属类别为1还是0
def classifyVector(P):
	prob=sigmoid(P)
	if prob>0.5:
		return 1.0
	else:
		return 0.0

#*****************************colicTest()与multiTest()属于对实例，可不看
def colicTest():
	frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
	trainingSet = []; trainingLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr =[]
		#print len(currLine)
		#print currLine
		for i in range(21):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[21]))
	trainWeights = stocGradAscent0(array(trainingSet), trainingLabels, 1000)
	errorCount = 0; numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec += 1.0
		currLine = line.strip().split('\t')
		lineArr =[]
		for i in range(21):
			lineArr.append(float(currLine[i]))
		if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
			errorCount += 1
	errorRate = (float(errorCount)/numTestVec)
	print "the error rate of this test is: %f" % errorRate
	return errorRate

def multiTest():
	numTests = 10; errorSum=0.0
	for k in range(numTests):
		errorSum += colicTest()
	print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))

#获取训练数据dataArr和标签labelMat
if __name__ == '__main__' :
    dataArr,labelMat=loadDataSet()
    
    '''
    #梯度上升 
    weights=gradAscent(dataArr, labelMat)
    plotBestFile(weights,dataArr,labelMat)
    '''
    
    #随机梯度上升
    weights1=stocGradAscent0(dataArr, labelMat,200)
    plotBestFile(weights1,dataArr,labelMat)
    
    #实际案例的应用可不看
    #multiTest()




