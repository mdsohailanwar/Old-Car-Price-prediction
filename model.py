import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor
#from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
import pickle
warnings.filterwarnings('ignore')

df = pd.read_csv(r'C:\Users\chandu\Documents\Board Infinity\Machine Learning\Projects\Capstone Project-2 (Old car price prediction)\car-data.csv')

df= df.drop_duplicates()

df['year'] = 2021 - df['year']

df.rename(columns = {'year':'age_of_car'}, inplace = True)

df = df.replace({'Automatic': 0, 'Manual': 1, 'Semi-Auto': 2})

df = df.replace({'Diesel': 0, 'Petrol': 1, 'Hybrid': 2, 'Other': 3})

x = df[df.engineSize == 0].index
df = df.drop(labels=x, axis =0)

X = df[['age_of_car','transmission','mileage','fuelType','tax','mpg','engineSize']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)


# ## RandomForestRegressor

reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)
y_pred = reg_rf.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

r2_score(y_test,y_pred)

r2 = (r2_score(y_test, y_pred))
n = X_test.shape[0] # Number of rows in out test data
p = X_test.shape[1] # Number of features in our data
adj_r2 = 1 - (((1-r2)*(n-1))/(n-p-1))
print("Adjusted R2 Error:", adj_r2)

# ## XGBRegressor

# XGB = XGBRegressor()
# XGB.fit(X_train, y_train)
# y_pred = XGB.predict(X_test)

# print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# print('MSE:', metrics.mean_squared_error(y_test, y_pred))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# r2 = (r2_score(y_test, y_pred))
# n = X_test.shape[0] # Number of rows in out test data
# p = X_test.shape[1] # Number of features in our data
# adj_r2 = 1 - (((1-r2)*(n-1))/(n-p-1))
# print("Adjusted R2 Error:", adj_r2)

f = open('reg_rf.pickle', 'wb')
pickle.dump(reg_rf, f)
f.close()


# f = open('XGB.pickle', 'wb')
# pickle.dump(reg_rf, f)
# f.close()