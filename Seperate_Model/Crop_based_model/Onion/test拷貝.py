import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns

data = pd.read_csv('/Users/kendergee/Desktop/India_Crop_Prediction/Seperate_CSV/Many_Crops/Khesari_data.csv')
X = data.drop(['Production','Crop'], axis=1)
y = data['Yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

OneHotColumns = ['Season', 'State']
encoder = OneHotEncoder(sparse_output=False, drop='first')

X_train_onehot = encoder.fit_transform(X_train[OneHotColumns])
X_test_onehot = encoder.transform(X_test[OneHotColumns])

X_train_onehot_df = pd.DataFrame(X_train_onehot, columns=encoder.get_feature_names_out(OneHotColumns))
X_test_onehot_df = pd.DataFrame(X_test_onehot, columns=encoder.get_feature_names_out(OneHotColumns))

X_train = pd.concat([X_train.drop(OneHotColumns, axis=1).reset_index(drop=True), X_train_onehot_df.reset_index(drop=True)], axis=1)
X_test = pd.concat([X_test.drop(OneHotColumns, axis=1).reset_index(drop=True), X_test_onehot_df.reset_index(drop=True)], axis=1)

sc = StandardScaler()
scColumns = ['Crop_Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

X_train[scColumns] = sc.fit_transform(X_train[scColumns])
X_test[scColumns] = sc.transform(X_test[scColumns])

from xgboost import XGBRegressor
model = XGBRegressor()

param_grid = {
'reg_alpha': [0, 0.01, 0.1, 1, 10],
'reg_lambda': [0, 0.01, 0.1, 1, 10],
'learning_rate': [0.01, 0.1],
'n_estimators': [100, 500],
'max_depth': [3, 4, 5]}

grid_search = GridSearchCV(estimator=model,param_grid=param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
grid_search.fit(X_train,y_train)
XGB_grid_MSE =-grid_search.best_score_
print("XGB: Best parameters found: ", grid_search.best_params_)
print("XGB: Best CV MSE: ", XGB_grid_MSE)

best_model = grid_search.best_estimator_
kfold_scores = cross_val_score(best_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
XGB_Kfold_MSE =-kfold_scores.mean()
print("XGB: Best model cross-validated MSE: {:.4f}".format(XGB_Kfold_MSE))

y_pred = best_model.predict(X_test)

XGB_test_mse = mean_squared_error(y_test, y_pred)
XGB_test_r2 = r2_score(y_test, y_pred)

print(f"XGB: Test set MSE: {XGB_test_mse}")
print(f"XGB: Test set R^2: {XGB_test_r2}")