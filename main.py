import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from math import ceil
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers

prices = pd.read_csv('gas.csv')
prices = prices[['Date', 'A1', 'R1', 'M1', 'P1']]
prices['week'] = prices.index
# prices['Date'] = pd.to_datetime(prices['Date'])
# print(prices.head())
# print(prices.tail())

x = prices[['week']].values
# change val here for type of gas to predict
y = prices[['A1']].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

last = datetime.strptime(prices['Date'].iloc[-1], '%m/%d/%Y')

week_now = int(ceil((datetime.today() - last).days/7))
print(week_now)

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylabel('error')
    plt.show()

def plot_preds(y_test, y_pred):
    plt.plot(y_test, label='true')
    plt.plot(y_pred, label='pred')
    plt.legend()
    plt.show()

def past_features(type):
    x_train = []
    y_train = []
    for i in range(16,len(x)):
        x_train.append(y[i-16:i])
        y_train.append(y[i])
    next_week = [y[len(y)-15:]]
    return np.array(x_train), np.array(y_train), np.array(next_week)

def dl_regression():
    # custom data analysis
    x_cust, y_cust, next_week = past_features('A1')
    x_train, y_train = x_cust[:int(len(x_cust)*0.85)], y_cust[:int(len(y_cust)*0.85)]
    x_test, y_test = x_cust[int(len(x_cust)*0.85):], y_cust[int(len(y_cust)*0.85):]
    model = keras.Sequential([layers.Dense(64, activation='relu'), layers.Dense(64, activation='relu'), layers.Dense(1)])
    model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(0.001))
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=100, verbose=0)
    plot_loss(history)
    losses = history.history['loss']
    print('Avg loss: ', sum(losses)/len(losses))
    y_pred = model.predict(x_test)
    # print(y_pred.mean(axis=1).shape)
    # print(y_test.shape)
    print('Test set: ', mean_squared_error(y_test, y_pred.mean(axis=1)))
    next_y = model.predict(next_week)
    print("next week: ", next_y.mean(axis=1))
   
    # plot_preds(y_test, y_pred)

def rnn_regression():
    x_cust, y_cust, next_week = past_features('A1')
    x_train, y_train = x_cust[:int(len(x_cust)*0.85)], y_cust[:int(len(y_cust)*0.85)]
    # print(x_train.shape)
    x_test, y_test = x_cust[int(len(x_cust)*0.85):], y_cust[int(len(y_cust)*0.85):]
    model = keras.Sequential([layers.LSTM(16, return_sequences=True), layers.Dropout(0.2), layers.LSTM(16), layers.Dense(1)])
    model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(0.001))
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=30, verbose=1)
    losses = history.history['loss']
    print('Avg loss: ', sum(losses)/len(losses))
    y_pred = model.predict(x_test)
    
    print('Test set: ', mean_squared_error(y_test, y_pred.mean(axis=1)))
    next_y = model.predict(next_week)
    print("next week: ", next_y.mean(axis=1))
    

def eval_models():
    models = []
    models.append(('svr', SVR()))
    models.append(('reg', LinearRegression()))
    models.append(('las', Lasso()))
    models.append(('dtr', DecisionTreeRegressor()))
    models.append(('eln', ElasticNet()))

    results = []
    best = 0
    best_n = ''
    for name, model in models:
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        res = cross_val_score(model, x_train, y_train, cv=kfold, scoring='neg_mean_absolute_error')
        results.append((name, res))
        if best < res.mean():
            best = res.mean()
            best_n = name
        print('{} : {}'.format(name, res.mean()))
    return best_n, best

def use_best():
    dtr = DecisionTreeRegressor()
    next_week = int(prices['week'].iloc[-1]) + week_now + 1
    dtr.fit(x_train, y_train)
    y_pred = dtr.predict(x_test)
    print(mean_squared_error(y_test, y_pred))
    pred = dtr.predict([[next_week]])

    print(pred)
    plt.plot(x, y)
    plt.plot(next_week, pred, 'ro')
    plt.show()

# dl_regression()
rnn_regression()
# best_model, best_score = eval_models()
# use_best()