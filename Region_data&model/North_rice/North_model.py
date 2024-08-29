import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import os
from sklearn.svm import SVR
import joblib
from sklearn.decomposition import PCA

# 獲取當前腳本所在的目錄
current_directory = os.path.dirname(os.path.abspath(__file__))

# 設定工作目錄為當前腳本所在的目錄
os.chdir(current_directory)


def preprocessing():
    data = pd.read_csv('North_rice.csv')
    data = data.dropna()
    X = data.drop(['Yield','Production'], axis=1)
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

    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def modeling(X_train, X_test, y_train, y_test):
    model = SVR(C= 100, epsilon= 0.1, kernel= 'rbf')
    model.fit(X_train,y_train)

    kfold_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
    XGBRegressor_Kfold_MSE =-kfold_scores.mean()
    print("Best model cross-validated MSE: {:.4f}".format(XGBRegressor_Kfold_MSE))

    y_pred = model.predict(X_test)
    XGBRegressor_test_mse = mean_squared_error(y_test, y_pred)
    XGBRegressor_test_r2 = r2_score(y_test, y_pred)

    print(f"XGBRegressor: Test set MSE: {XGBRegressor_test_mse}")
    print(f"XGBRegressor: Test set R^2: {XGBRegressor_test_r2}")
    
    return model

def save_file(model):
    joblib.dump(model,'North_model.pkl')


X_train, X_test, y_train, y_test = preprocessing()
model = modeling(X_train,X_test,y_train,y_test)
save_file(model)
