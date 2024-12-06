import numpy as np
import matplotlib.pyplot as plt
import math as math
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
# def Create_Data(n): #创建随机的训练集，以一定规则分类
#     D = np.random.randint(0, 100, size=[n, 3])
#     D_list = D.tolist()
#     for each in D_list:
#         if each[-2]**2 + each[-3] >= 1000:
#             each[-1] = 1
#             continue
#         if each[-2]**2 + each[-3] < 1000:
#             each[-1] = 0
#     return D_list

# def Save_List(list, filename):
#     with open(filename, "w") as file:
#         for item in list:
#             file.write(str(item) + "\n")

def loadDataSet(filename):   #读取数据（这里只有两个特征）
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])   #前面的1，表示方程的常量。比如两个特征X1,X2，共需要三个参数，W1+W2*X1+W3*X2
        labelMat.append(int(lineArr[-1]))
    return dataMat,labelMat

def Classify_Data(mat_x,mat_y): #将训练集按正负类拆开
    positive_list=[]
    negative_list=[]
    m=len(mat_y)
    for i in range(m):
        if mat_y[i,0] == 0:
            negative_list.append(mat_x[i,:])
        if mat_y[i,0] == 1:
            positive_list.append(mat_x[i,:])
    return positive_list,negative_list

def Each_Column(list):     #分割提取特征空间x与分类标签y
    column_1 = [row[0:1] for row in list]  # 取数据集前两列
    column_2 = [row[1:2] for row in list]
    column_3 = [row[-1:] for row in list]
    return column_2[1]

def Get_h(arr_w,mat_x):
    inner=mat_x*arr_w
    N=len(inner)
    h=np.zeros(N)
    for i in range(N):
        if abs(inner[i])<=10:
            h[i]=1/(1+math.e**(-float(inner[i])))
        if inner[i]>10:
            h[i]=1-10**(-10)
        if inner[i] < -10:
            h[i] = 10**(-10)
    h = np.mat(h).T
    return h

def Log_Appro(y,h,mat_xi,mat_u,Lambda,rho,mat_w):
    N=len(y)
    L=0
    for i in range(N):
        L=L+(y[i]*math.log(h[i])+(1-y[i])*math.log(1-h[i]))
    L=(-1/N)*L+Lambda*np.linalg.norm(mat_xi,1)+0.5*rho*(np.linalg.norm(mat_w-mat_xi,2)**2)+mat_u.T * (mat_w - mat_xi)
    return L


def Gradient_Decrease( mat_x, mat_w, alpha,mat_xi,mat_u,Lambda,rho):
    h = Get_h(mat_w, mat_x)
    lg=[]
    tt=0
    for i in range(1000):
        lg1 = Log_Appro(mat_y, h,mat_xi,mat_u,Lambda,rho,mat_w)
        h = Get_h(mat_w, mat_x)
        error = mat_y - h
        mat_w = mat_w + alpha * (mat_x.T * error + Lambda+mat_u+rho*(mat_w-mat_xi))
        lg2 = Log_Appro(mat_y, h,mat_xi,mat_u,Lambda,rho,mat_w)
        lg.append(lg2)
        tt=tt+1
        if i > 2:
            if lg1 - lg2 < 1 * 10 ** (-8):
                break
    return mat_w,lg,tt

def Soft_Liminail(Lambda,rho,mat_u,mat_w):
    N,n=np.shape(mat_u)
    S_Lr=np.ones((N,1))
    for i in range(N):
        if ((mat_u[i])+mat_w[i])>Lambda/rho:
            S_Lr[i]=(mat_u[i])+mat_w[i]-Lambda/rho
        if abs(((mat_u[i]) + mat_w[i])) <= Lambda / rho:
            S_Lr[i]=0
        if ((mat_u[i]) + mat_w[i]) < -Lambda / rho:
            S_Lr[i]=(mat_u[i])+mat_w[i]+Lambda/rho
    return S_Lr

# def Output_Picture(x_1,x_2,X_1,X_2): #画出训练集的分布图
#     plt. figure(figsize=(10,10), dpi=100)
#     plt.scatter(x_1, x_2, color='r')
#     plt.scatter(X_1, X_2, color='b')
#     plt.show()
#------------------------------------------------------------------
# n=20
# Test_Data=Create_Data(n)
# Save_List(Test_Data,"Data.txt")
mat_x,mat_y=loadDataSet('Data.txt')
mat_x=np.mat(mat_x)
m,n = np.shape(mat_x)
mat_y=np.mat(mat_y).T
for i in range(m):
    if mat_y[i]==2:
        mat_y[i]=0
mat_w=np.ones((n,1))
mat_xi=np.ones((n,1))
mat_u=np.ones((n,1))
Lambda=0.2
rho=0.1
alpha=0.001

# #------------------------------------------------------------------
#
# #~~~~~~~~~~~~~~~~~~~~~~~~结果验证板块~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
lg=0
timee=0
for i in range(105):
    # temp_lg=lg
    [mat_w, lg,tt] = Gradient_Decrease(mat_x, mat_w, alpha, mat_xi, mat_u, Lambda, rho)
    timee=timee+tt
    # epsilon=temp_lg-lg
    mat_xi=Soft_Liminail(Lambda,rho,mat_u,mat_w)
    mat_u=mat_u+(mat_w-mat_xi)
h=Get_h(mat_w,mat_x)
mat_ans=np.ones((m,1))
for i in range(m):
    if h[i]<=0.5:
        mat_ans[i]=0
    if h[i]>0.5:
        mat_ans[i]=1
check=mat_ans-mat_y
mistake=0
for i in range(m):
    if check[i]!=0:
        mistake=mistake+1
print('拟合错误率')
print(mistake/m)

mat_x1,mat_y1=loadDataSet('Data1.txt')
mat_x1=np.mat(mat_x1)
m1,n1 = np.shape(mat_x1)
mat_y1=np.mat(mat_y1).T
h1=Get_h(mat_w,mat_x1)
mat_ans1=np.ones((m1,1))
for i in range(m1):
    if h1[i]<=0.5:
        mat_ans1[i]=0
    if h1[i]>0.5:
        mat_ans1[i]=1
check1=mat_ans1-mat_y1
mistake1=0
for i in range(m1):
    if check1[i]!=0:
        mistake1=mistake1+1
print('测试集错误率')
print(mistake1/m1)
print("对权重迭代次数")
print(timee)
print('输出权重')
print(mat_w)
print("按任意键推出程序")
input()