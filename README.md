# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## Aim:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import pandas, numpy and sklearn.
2.Calculate the values for the training data set.
3.Calculate the values for the test data set.
4.Plot the graph for both the data sets and calculate for MAE, MSE and RMSE. 
 
## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: JANANI.S 
RegisterNumber: 212222230049 
```
```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('/student_scores.csv')
#Displying the contents in datafile
df.head()

df.tail()

#Segregating data to variables
X=df.iloc[:,:-1].values
X

Y=df.iloc[:,-1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying the predicted values
Y_pred

#displaying the actual values
Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="blue")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="gold")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE= ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE= ",mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
### Contents in the data file
#### df.head()
![head](https://github.com/JananiSoundararajan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477549/047cb0f2-de48-4c12-9ca4-5908f047d9ef)
#### df.tail()
![tail](https://github.com/JananiSoundararajan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477549/e47c1008-991b-45a7-abbf-938b0bb29c37)

### X and Y datasets from original dataset:
![X](https://github.com/JananiSoundararajan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477549/295de532-515f-4480-a606-877685206481)
![Y](https://github.com/JananiSoundararajan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477549/41b17b11-5c0b-45a0-be84-493fb8fdad53)
### Values of Y prediction and array value of Y test:
![YPRED](https://github.com/JananiSoundararajan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477549/dfdc409b-a280-4783-8d5b-dc51240df145)
![YTEST](https://github.com/JananiSoundararajan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477549/c76dc1cc-5d1b-4c70-b96a-741cb1be9d2b)
### Training set graph:
![TNS](https://github.com/JananiSoundararajan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477549/b5085730-cfef-49a9-84f5-40660c2091db)
### Test set graph:
![TTS](https://github.com/JananiSoundararajan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477549/6fab3fff-2cf8-4e5d-9475-94134bfe766c)
### Values of MSE,MAE and RMSE:
![ANS](https://github.com/JananiSoundararajan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477549/cdc65d42-e863-45d8-82ff-d28e09627989)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
