# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: Avanesh.R
RegisterNumber:25018356
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    x = np.c_[np.ones(len(X1)), X1]  # Add intercept term to X1
    theta = np.zeros(x.shape[1]).reshape(-1, 1)  # Initialize theta
    for _ in range(num_iters):
        predictions = x.dot(theta).reshape(-1, 1)  # Predictions
        errors = predictions - y  # Errors
        theta -= learning_rate * (1 / len(X1)) * x.T.dot(errors)  # Update theta
    return theta

try:
    # Attempt to load the dataset
    data = pd.read_csv('50_Startups.csv', header=None)
except FileNotFoundError:
    print("Error: The file '50_Startups.csv' was not found.")
else:
    X = data.iloc[1:, :-2].values  # Independent variables
    X1 = X.astype(float)
    scaler = StandardScaler()
    y = data.iloc[1:, -1].values.reshape(-1, 1)  # Dependent variable (target)
    
    X1_Scaled = scaler.fit_transform(X1)  # Scale independent variables
    Y1_Scaled = scaler.fit_transform(y)  # Scale target variable
    
    theta = linear_regression(X1_Scaled, Y1_Scaled)  # Train linear regression
    
    # New data for prediction
    new_data = np.array([165349.2, 136897.8, 471784.1]).reshape(1, -1)
    new_Scaled = scaler.transform(new_data)  # Scale new data
    prediction = np.dot(np.append(1, new_Scaled), theta)  # Predict using the model
    pre = scaler.inverse_transform(prediction.reshape(1, -1))  # Inverse scale the prediction
    
    print(f"Prediction value: {pre}")

```

## Output:
![imageex ml](https://github.com/user-attachments/assets/41b339a2-e2d0-4b7c-bb67-528c6ca19a67)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
