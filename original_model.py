# initial model - not good!
# trains SVR model on VIs vs GDD -> GDD to Pollen

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# read in csv that has VIs, flowering, and stand count data
data = pd.read_csv('MI23 and FL22 VI Accession Flowering.csv')

# column names for reference
# id,Date,Accumulated.GDD,NDVI,NDRE,EVI,CI,RTVIcore,SAVI,MSAVI,Accession,Stand Count,FL GDD to 50% Pollen,FL GDD to 50% Silk

# remove columns don't want in training
X = data.drop(columns=['FL GDD to 50% Silk', 'Date', 'id', 'Accession', 'FL GDD to 50% Pollen'])
#print(X)

# set the target value
y = data['FL GDD to 50% Pollen']

# split data into test and train
# random_state gives same random output each time the code is run
# test_size sets the % of data kept for test
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=105, test_size=0.20)

# scale the data so larger numbers don't impact output more
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# set up support vector regression model
svr_rbf = SVR(kernel="rbf")

# train on scaled data
svr_rbf.fit(X_train_scaled, y_train)

# predict on scaled test data
y_pred = svr_rbf.predict(X_test_scaled)


# code for training and testing without scaling
# svr_rbf.fit(X_train, y_train)
# y_pred = svr_rbf.predict(X_test)

# print root mean squared error and R^2
print('RMSE', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R^2', r2_score(y_test, y_pred))

# plot predicted vs. actual
plt.scatter(y_pred, y_test, color='green')

# calc line of best fit to display
a, b = np.polyfit(y_pred, y_test, 1)
plt.plot(y_pred, a*y_pred+b)

# plot labels
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Predicting GDD to 50% Pollen with SVR Model')

# save plot
plt.savefig('Original Output.png', dpi=360)

# show plot
plt.show()