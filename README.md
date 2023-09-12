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
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()

#segregating data to variables
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![image](https://github.com/shalini-venkatesan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118720291/e9c9af3b-e6d7-4c38-8287-2d39734b8a13)
![image](https://github.com/shalini-venkatesan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118720291/b0f6557c-61fc-4fda-abf7-111c22d9e568)
### Array value of X :
![image](https://github.com/shalini-venkatesan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118720291/9dde2cd7-bc53-4abd-8860-7dc31b32a78b)
### Array value of Y : 
![image](https://github.com/shalini-venkatesan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118720291/f7881ace-a40f-409e-809c-5a489b5a05b6)
### Values of Y prediction :
![image](https://github.com/shalini-venkatesan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118720291/ccb96796-4d77-400a-80c7-28b96581efed)
### Values of Y test :
![image](https://github.com/shalini-venkatesan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118720291/0f2b45a4-0f27-4a5f-a7f1-651ca5e66523)
### Training Set Graph :
![image](https://github.com/shalini-venkatesan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118720291/3edadd19-68da-4968-8117-c4d80dee58fc)
### Test Set Graph :
![image](https://github.com/shalini-venkatesan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118720291/8e0481e5-9703-4f9b-aba2-309caed8f2fe)
### Values of MSE, MAE and RMSE :
![image](https://github.com/shalini-venkatesan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118720291/0d9b427b-769e-4268-8c72-a9441adbdbec)
![image](https://github.com/shalini-venkatesan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118720291/a1fe8358-8d96-448b-b55f-fc516d04a775)
![image](https://github.com/shalini-venkatesan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118720291/94513471-9dcb-4147-9f82-b3cb3e42b541)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
