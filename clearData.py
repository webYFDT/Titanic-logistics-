#coding:utf-8
from sklearn.ensemble import RandomForestRegressor
import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
import sklearn.preprocessing as preprocessing
import math
### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]
    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)

    rfr.fit(X, y)
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df
def FareGetValue(x):
    if math.isnan(x):
        x=35.627188             ###此处取Fare平均值，需要改进
    return x
data_train = pd.read_csv("test.csv")
data_train['Fare']=data_train.Fare.map(FareGetValue)##***********只有test数据缺一个Fare值，用平均值填充
data_train,rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)


scaler = preprocessing.StandardScaler()
#age_scale_param = scaler.fit(df['Age'].reshape(1,-1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1))
#fare_scale_param = scaler.fit(df['Fare'].reshape(1,-1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1))
df.drop(['Age', 'Fare'], axis=1, inplace=True)
#print df
df.to_csv('NewTest.csv')