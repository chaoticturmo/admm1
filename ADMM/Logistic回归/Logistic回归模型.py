import numpy as np
import math as math
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=RuntimeWarning)
def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]),float(lineArr[2]),float(lineArr[5]),float(lineArr[6]),float(lineArr[10]),float(lineArr[14])])  #前面的1，表示方程的常量。比如两个特征X1,X2，共需要三个参数，W1+W2*X1+W3*X2
        labelMat.append(int(lineArr[-1]))
    return dataMat,labelMat

def Sigmoid(mat_w,mat_x):
    inn=mat_x*mat_w
    [m,n]=np.shape(inn)
    sim=np.ones((m,1))
    for i in range(m):
        if inn[i,0]>10:
            sim[i]=0.99999
        if inn[i,0]<=10:
            sim[i]=(math.e ** (inn[i,0])) / (1 + math.e ** (inn[i,0]))
    return sim

def Gradient(mat_x,mat_y,sim,Lambda):
    [m,n]=np.shape(mat_x)
    g_w = np.ones((n, 1))
    for j in range (n):
        sum=0
        for i in range (m):
            sum=sum+(sim[i]-mat_y[i])*mat_x[i,j]
        g_w[j]=sum+Lambda
    return g_w

#----------------------------训练集导入---------------------------------------
mat_x,mat_y=loadDataSet('Data000.txt')
mat_xc=[row[1:] for row in mat_x]
mat_x=np.mat(mat_x)
mat_xc=np.mat(mat_xc)
m,n = np.shape(mat_x)
m,p=np.shape(mat_xc)
mat_y=np.mat(mat_y).T
for i in range(m):
    if mat_y[i]==2:
        mat_y[i]=0

#-----------------------------数据初始化--------------------------------------
mat_w=np.zeros((p+1,1))
Lambda=0.2
sim=Sigmoid(mat_w,mat_x)
mat_I=np.ones((m,1))
alpha=0.00003
# #-----------------------------学习区域---------------------------------------
for i in range (300):
    g_w=Gradient(mat_x,mat_y,sim,Lambda)
    mat_w=mat_w-alpha*g_w
    sim=Sigmoid(mat_w,mat_x)
Train_result=Sigmoid(mat_w,mat_x)
count = 0
for i in range(m):
    if Train_result[i]<0.6:
        Train_result[i]=0
    else:
        Train_result[i]=1

for i in range(m):
    if (Train_result[i]-mat_y[i])==0:
        count = count+1
print('拟合准确率')
print(count/m)
#-------------------------测试区域---------------------------------------
mat_x1,mat_y1=loadDataSet('Data001.txt')
mat_x1=np.mat(mat_x1)
[mm,nn]=np.shape(mat_x1)
mat_y1=np.mat(mat_y1).T
for i in range(mm):
    if mat_y1[i]==2:
        mat_y1[i]=0
Train_result1=Sigmoid(mat_w,mat_x1)
count1 = 0
for i in range(mm):
    if Train_result1[i]<=0.6:
        Train_result1[i]=0
    else:
        Train_result1[i]=1

for i in range(mm):
    if (Train_result1[i]-mat_y1[i])==0:
        count1 = count1+1
print('测试集准确率')
print(count1/mm)
print('最终权重')
print(mat_w)
print('按任意键推出程序')
input()
