import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def preprocessing():
    data = pd.read_csv('Seperate_Model/Crop_based_model/Rice/West_rice/West_rice.csv')
    data = data.dropna()
    X = data.drop('Yield',axis=1)
    y = data['Yield']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    OneHotColumns = ['Season', 'State']
    encoder = OneHotEncoder(sparse_output=False, drop='first')

    X_train_onehot = encoder.fit_transform(X_train[OneHotColumns])
    X_test_onehot = encoder.transform(X_test[OneHotColumns])

    X_train_onehot_df = pd.DataFrame(X_train_onehot, columns=encoder.get_feature_names_out(OneHotColumns))
    X_test_onehot_df = pd.DataFrame(X_test_onehot, columns=encoder.get_feature_names_out(OneHotColumns))

    X_train = pd.concat([X_train.drop(OneHotColumns, axis=1).reset_index(drop=True), X_train_onehot_df.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.drop(OneHotColumns, axis=1).reset_index(drop=True), X_test_onehot_df.reset_index(drop=True)], axis=1)

    sc = StandardScaler()
    scColumns = ['Crop_Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide','Production']

    X_train[scColumns] = sc.fit_transform(X_train[scColumns])
    X_test[scColumns] = sc.transform(X_test[scColumns])

    return X_train,X_test,y_train,y_test

def ann(X_train,y_train):
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
    ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
    ann.add(tf.keras.layers.Dense(units=1))

    ann.compile(optimizer='adam',loss='mean_squared_error')
    ann.fit(X_train,y_train,batch_size=32,epochs=300)

    return ann

def evaluate(ann,X_test,y_test):
    y_pred = ann.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    print(mse)


X_train,X_test,y_train,y_test = preprocessing()
ann = ann(X_train,y_train)
evaluate(ann,X_test,y_test)