import numpy as np
import math as math
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=RuntimeWarning)
def loadDataSet(filename):   #读取数据（这里只有两个特征）
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]),float(lineArr[2]),float(lineArr[5]),float(lineArr[6]),float(lineArr[9]),float(lineArr[10]),float(lineArr[14])])  #前面的1，表示方程的常量。比如两个特征X1,X2，共需要三个参数，W1+W2*X1+W3*X2
        labelMat.append(int(lineArr[-1]))
    return dataMat,labelMat

def Simgoid(mat_w,mat_x):
    inn=mat_x*mat_w
    [m,n]=np.shape(inn)
    sim=np.ones((m,1))
    for i in range(m):
        if inn[i,0]>10:
            sim[i]=0.99999
        if inn[i,0]<=10:
            sim[i]=(math.e ** (inn[i,0])) / (1 + math.e ** (inn[i,0]))
    return sim

def Mat_Omiga(sim):
    [m,n]=np.shape(sim)
    list_sim=[]
    for i in range(m):
        list_sim.append(sim[i,0]*(1-sim[i,0]))
    mat_omiga=np.diag(list_sim)
    return mat_omiga

def Lambda_K(Lambda,mat_x,mat_omiga):
    [n,k]=np.shape(mat_x)
    Lambda_k=np.ones((k,1))
    for j in range(k):
        xigema=0
        for i in range(n):
            xigema=xigema+mat_omiga[i,i]*(mat_x[i,j])**2
        Lambda_k[j,0]=Lambda/xigema
    return Lambda_k

def Z(mat_xc,mat_w,mat_omiga,mat_y,sim):
    return mat_xc*mat_w+np.linalg.inv(mat_omiga)*(mat_y-sim)

def W_0(mat_omiga,mat_z,mat_xc,mat_wc):
    [m,n]=np.shape(mat_omiga)
    [m,p]=np.shape(mat_xc)
    lower=0
    for i in range(m):
        lower=lower+mat_omiga[i,i]
    upper=0
    for ii in range(m):
        sum_xw=0
        for jj in range(p):
            sum_xw=sum_xw+mat_xc[ii,jj]*mat_wc[jj]
        upper=upper+mat_omiga[ii,ii]*(mat_z[ii,0]-sum_xw)
    w_0=upper/lower
    return w_0

def C_K(mat_omiga,mat_z,w_0,mat_xc,mat_wc):
    [n,N]=np.shape(mat_omiga)
    [n,p]=np.shape(mat_xc)
    c_k = np.ones((p, 1))
    for k in range(p):
        upper=0
        for i in range(n):
            sum_xw=0
            for j in range(p):
                if j!=k:
                    sum_xw=sum_xw+mat_xc[i,j]*mat_wc[j,0]
            upper=upper+mat_omiga[i,i]*(mat_z[i,0]-w_0-sum_xw)*mat_xc[i,k]
        lower=0
        for ii in range(n):
            lower=lower+mat_omiga[ii,ii]*(mat_xc[i,k])**2
        c_k[k]=upper/lower
    return c_k

def Phi(Lambda_k,c_k):
    [k,K]=np.shape(c_k)
    phi=np.ones((k, 1))
    for i in range(k):
        phi[i]=math.acos((Lambda_k[i]/8)*(c_k[i]/3)**(-1.5))
    return phi

def W_K(c_k,phi,Lambda_k):
    [n,N]=np.shape(c_k)
    w_k=np.ones((n, 1))
    for i in range(n):
        if abs(c_k[i])>=(-3/4)*(Lambda_k[i])**(2/3):
            w_k[i]=(2/3)*c_k[i]*(1+math.cos((2/3)*math.pi-(2/3)*phi[i]))
        else:
            w_k[i]=0
    return w_k

def Refresh_Z(mat_x,best_w,mat_y,Train_result,mat_omiga):
    [n,N]=np.shape(mat_x)
    z=np.ones((n,1))
    for i in range(n):
        z[i]=mat_x[i]*best_w+(mat_y[i,0]-Train_result[i,0])/mat_omiga[i,i]
    return z

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
mat_wc=np.zeros((p,1))
Lambda=0.2
sim=Simgoid(mat_wc,mat_xc)
mat_omiga=Mat_Omiga(sim)
mat_z=Z(mat_xc,mat_wc,mat_omiga,mat_y,sim)
Lambda_k=Lambda_K(Lambda,mat_xc,mat_omiga)
mat_I=np.ones((m,1))
# #-----------------------------学习区域---------------------------------------
for ii in range(9):
    w_0=W_0(mat_omiga,mat_z,mat_xc,mat_wc)
    c_k=C_K(mat_omiga,mat_z,w_0,mat_xc,mat_wc)
    phi=Phi(Lambda_k,c_k)
    Lambda_k = Lambda_K(Lambda, mat_xc, mat_omiga)
    w_k=W_K(c_k,phi,Lambda_k)
    best_w=np.ones((n,1))
    for i in range(n):
        if i==0:
            best_w[i]=w_0
        else:
            best_w[i]=w_k[i-1]
    Train_result=Simgoid(best_w,mat_x)
    mat_wc=w_k
    mat_omiga=Mat_Omiga(Train_result*(mat_I-Train_result))
    mat_z=Refresh_Z(mat_x,best_w,mat_y,Train_result,mat_omiga)
count = 0
for i in range(m):
    if Train_result[i]<0.5:
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
Train_result1=Simgoid(best_w,mat_x1)
count1 = 0
for i in range(mm):
    if Train_result1[i]<=0.5:
        Train_result1[i]=0
    else:
        Train_result1[i]=1

for i in range(mm):
    if (Train_result1[i]-mat_y1[i])==0:
        count1 = count1+1
print('测试集准确率')
print(count1/mm)
print('最终权重')
print(best_w)
print('按任意键推出程序')
input()