# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas library to read csv or excel file.
2.Import LabelEncoder using sklearn.preprocessing library.
3.Transform the data's using LabelEncoder.
4.Import decision tree classifier from sklearn.tree library to predict the values.
5.Find accuracy.
6.Predict the values.
7.End of the program.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: NIRAUNJANA GAYATHRI G R
RegisterNumber:  22008369
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:,[a0,1]]
y=data[:,2]

print("Array of X") 
X[:5]

print("Array of y") 
y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
print("Exam 1- score Graph")
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))
    
 plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
print("Sigmoid function graph")
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad
    
 X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print("X_train_grad value")
print(J)
print(grad)


X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print("Y_train_grad value")
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad
    
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(" Print res.x")
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()

print("Decision boundary - graph for exam score")
plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1, 45, 85]),res.x))
print("Proability value ")
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)

print("Prediction value of mean")
np.mean(predict(res.x,X) == y)

```

## Output:

![image](https://user-images.githubusercontent.com/119395610/235346278-d394502d-9c81-495c-ab89-71a61c1a182f.png)

![image](https://user-images.githubusercontent.com/119395610/235346284-cdc710e0-bb19-4b28-a6f5-07914ead053a.png)

![image](https://user-images.githubusercontent.com/119395610/235346292-ee3a20b9-7559-4997-85fb-3b5ab889024f.png)


![image](https://user-images.githubusercontent.com/119395610/235346302-feef0ab5-ee69-4a50-8f2d-6e88ec8fbf0f.png)

![image](https://user-images.githubusercontent.com/119395610/235346310-39c6bc91-9e82-48ae-b5d3-98dd07da106e.png)

![image](https://user-images.githubusercontent.com/119395610/235346321-ad0c0317-f510-4c0f-8b79-aedfc5a4903c.png)

![image](https://user-images.githubusercontent.com/119395610/235346331-e38d38c4-eac9-4fc0-8924-85dac42c4259.png)

![image](https://user-images.githubusercontent.com/119395610/235346344-8959fdf3-8519-4369-a10e-642b9319dd4a.png)

![image](https://user-images.githubusercontent.com/119395610/235346355-a7a7392c-3f09-4e36-a319-4d3f198e21ce.png)

![image](https://user-images.githubusercontent.com/119395610/235346367-4d53df44-3532-4561-8c0d-1b15f1a31bab.png)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

