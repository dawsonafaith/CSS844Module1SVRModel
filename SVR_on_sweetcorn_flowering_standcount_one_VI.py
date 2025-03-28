# train SVR model on one VI at a time

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# read in csv that has VIs, flowering, and stand count data
data = pd.read_csv('MI23 and FL22 VI Accession Flowering StandCount.csv')

# remove first & last date so only looking at middle 3
data = data[data.Date != 7.12]
data = data[data.Date != 10.02]

data = data[data['FL GDD to 50% Pollen'] < 1700]
data = data[data['FL GDD to 50% Pollen'] > 1000]
# print(data)

# column names for reference
# id,Date,Accumulated.GDD,NDVI,NDRE,EVI,CI,RTVIcore,SAVI,MSAVI,Accession,Stand Count,FL GDD to 50% Pollen,FL GDD to 50% Silk


# try model with just one VI one at a time
VIs = ['NDVI','NDRE','EVI','CI','RTVIcore','SAVI','MSAVI']
for vi in VIs:
    # only want the VI, GDD, and Stand Count for input to model
    X = data[[vi, 'Accumulated.GDD', 'Stand Count']].copy()
    # print(X)

    # set target value
    y = data['FL GDD to 50% Pollen']

    # split data into test and train
    # random_state gives same random output each time the code is run
    # test_size sets the % of data kept for test
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=105, test_size=0.15)

    # scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # set up and train model
    svr_rbf = SVR(kernel="rbf", epsilon=50, C=50, gamma='auto')
    svr_rbf.fit(X_train_scaled, y_train)

    # predict based on trained model
    y_pred = svr_rbf.predict(X_test_scaled)

    print('\n' + vi + ':')
    print('RMSE', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('R^2', r2_score(y_test, y_pred))

    plt.scatter(y_pred, y_test, color='green')
    a, b = np.polyfit(y_pred, y_test, 1)
    plt.plot(y_pred, a*y_pred+b)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Model Trained on GDD, Stand Count, and ' + vi)
    x_text_loc = min(y_pred) + 5
    plt.text(x_text_loc, 1680, 'R$^2$ = ' + str(r2_score(y_test, y_pred))[:5], fontsize=12)
    plt.savefig('model trained on ' + vi + '.png', dpi=360)
    plt.show()
