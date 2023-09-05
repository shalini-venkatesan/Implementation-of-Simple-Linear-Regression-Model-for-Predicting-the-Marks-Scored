# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. To implement the linear regression using the standard libraries in the python.
2. Use slicing function() for the x,y values.
3. Using sklearn library import training , testing and linear regression modules.
4. Predict the value for the y.
5. Using matplotlib library plot the graphs.
6. Use xlabel for hours and ylabel for scores.
7. End the porgram.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Shalini Venkatesan
RegisterNumber: 212222240096


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('/content/student_scores - student_scores.csv')
df.head()
X=df.iloc[:,:-1].values
X
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
y_pred
y_test
plt.scatter(X_train,y_train,color='violet')
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("h vs s (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,y_test,color='purple')
plt.plot(X_test,regressor.predict(X_test),color="black")
plt.title("h vs s (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

```

## Output:
![image](https://github.com/shalini-venkatesan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118720291/b26be599-d3c3-42a9-abfd-0e935e9df1fa)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
