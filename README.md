# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: avanesh r
RegisterNumber:  25018356
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression (X1, y):
    X = np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    num_iters = 1000   # number of iterations
    learning_rate = 0.01
    for _ in range(num_iters):
       predictions = X.dot(theta).reshape(-1, 1)
       errors = (predictions - y).reshape(-1, 1)
       theta = theta - learning_rate * (1 / len(X)) * X.T.dot(errors)
       return thet
from google.colab import drive
drive.mount('/content/drive')

data=pd.read_csv("/content/drive/MyDrive/50_Startups.csv")
data.head(11)

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)

print("X =",X)
print("X1_Scaled =",X1_Scaled)

theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data= np.array([165349.2, 136897.8, 471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_scaled), theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform (prediction)
print("prediction =",prediction)
print(f"Predicted value: {pre}")


```

## Output:
<img width="669" height="424" alt="Screenshot 2025-12-11 092448" src="https://github.com/user-attachments/assets/bc023d5c-8986-43d9-babc-699bf27d5815" />


<img width="377" height="724" alt="Screenshot 2025-12-11 092507" src="https://github.com/user-attachments/assets/1ff5a77c-4fef-4bb1-8279-060b3bf2e25d" />


<img width="612" height="743" alt="Screenshot 2025-12-11 092518" src="https://github.com/user-attachments/assets/3e0d3946-4db4-4ca5-a9d3-17f510a0be7c" />



<img width="363" height="48" alt="Screenshot 2025-12-11 092525" src="https://github.com/user-attachments/assets/436ae49e-c578-4d37-9b5c-f3ecad64c563" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
