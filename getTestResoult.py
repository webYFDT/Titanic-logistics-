# -*- coding: utf-8 -*-
import pandas as pd #数据分析
import numpy as np #科学计算
from logRegres import  stocGradAscent0,classifyVector
train_data=pd.read_csv("NewTrain.csv")
test_data=pd.read_csv("NewTest.csv")

train_data_xy=train_data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_data_xy_np=np.array(train_data_xy)

TrainY=train_data_xy_np[:,0]
TrainX=train_data_xy_np[:,1:]
TrainLen=len(TrainX)
one=np.ones(TrainLen)
NewTrainX=np.column_stack((one,TrainX))

test_x_np=test_data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
testX=np.array(test_x_np)
TestLen=len(testX)
oneTest=np.ones(TestLen)
NewTestX=np.column_stack((oneTest,testX))

w=stocGradAscent0(NewTrainX,TrainY,10000)

P=NewTestX.dot(w)  #test数据值
PP=pd.DataFrame({'Survived':P})
testY=PP.Survived.map(classifyVector)
testY.to_csv('resoult10000.csv',header=True)
    
