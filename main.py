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

prices = pd.read_csv('gas.csv')
prices = prices[['Date', 'A1', 'R1', 'M1', 'P1']]
prices['week'] = prices.index
# prices['Date'] = pd.to_datetime(prices['Date'])
# print(prices.head())
# print(prices.tail())

x = prices[['week']].values
y = prices[['P1']].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

last = datetime.strptime(prices['Date'].iloc[-1], '%m/%d/%Y')

week_now = int(ceil((datetime.today() - last).days/7))
print(week_now)


def eval_models():
    models = []
    models.append(('svr', SVR()))
    models.append(('reg', LinearRegression()))
    models.append(('las', Lasso()))
    models.append(('dtr', DecisionTreeRegressor()))
    models.append(('eln', ElasticNet()))

    results = []
    for name, model in models:
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        res = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
        results.append((name, res))
        print('{} : {}'.format(name, res.mean()))

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

# eval_models()
use_best()