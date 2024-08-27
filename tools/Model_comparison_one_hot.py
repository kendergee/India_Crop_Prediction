import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns


def preprocessing():
    data = pd.read_csv('/Users/kendergee/Desktop/vscode/India_Crop_Prediction/Rice/Central_rice/Central_rice.csv')
    X = data.drop('Production', axis=1)
    y = data['Production']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    OneHotColumns = ['Season', 'State']
    encoder = OneHotEncoder(sparse_output=False, drop='first')

    X_train_onehot = encoder.fit_transform(X_train[OneHotColumns])
    X_test_onehot = encoder.transform(X_test[OneHotColumns])

    X_train_onehot_df = pd.DataFrame(X_train_onehot, columns=encoder.get_feature_names_out(OneHotColumns))
    X_test_onehot_df = pd.DataFrame(X_test_onehot, columns=encoder.get_feature_names_out(OneHotColumns))

    X_train = pd.concat([X_train.drop(OneHotColumns + ['Yield'], axis=1).reset_index(drop=True), X_train_onehot_df.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.drop(OneHotColumns + ['Yield'], axis=1).reset_index(drop=True), X_test_onehot_df.reset_index(drop=True)], axis=1)
    
    sc = StandardScaler()
    scColumns = ['Crop_Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

    X_train[scColumns] = sc.fit_transform(X_train[scColumns])
    X_test[scColumns] = sc.transform(X_test[scColumns])
    
    return X_train, X_test, y_train, y_test

def ElasticNet(X_train, X_test, y_train, y_test):
    def find_best_params(X_train,y_train):
        from sklearn.linear_model import ElasticNet
        model = ElasticNet()
    
        param_grid = {
            'alpha': [0.01, 0.1, 1, 10, 100],
            'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1]}

        grid_search = GridSearchCV(estimator=model,param_grid=param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
        grid_search.fit(X_train,y_train)
        Elastic_grid_MSE =-grid_search.best_score_
        print("Elastic: Best parameters found: ", grid_search.best_params_)
        print("Elastic: Best CV MSE: ", Elastic_grid_MSE)

        best_model = grid_search.best_estimator_
        kfold_scores = cross_val_score(best_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
        Elastic_Kfold_MSE =-kfold_scores.mean()
        print("Elastic: Best model cross-validated MSE: {:.4f}".format(Elastic_Kfold_MSE))

        return best_model,Elastic_grid_MSE,Elastic_Kfold_MSE
    
    def evaluate_on_test_set(best_model,X_test,y_test):
        y_pred = best_model.predict(X_test)

        Elastic_test_mse = mean_squared_error(y_test, y_pred)
        Elastic_test_r2 = r2_score(y_test, y_pred)

        print(f"Elastic: Test set MSE: {Elastic_test_mse}")
        print(f"Elastic: Test set R^2: {Elastic_test_r2}")

        return Elastic_test_mse,Elastic_test_r2

    best_model,Elastic_grid_MSE,Elastic_Kfold_MSE = find_best_params(X_train,y_train)
    Elastic_test_mse,Elastic_test_r2 = evaluate_on_test_set(best_model,X_test,y_test)

    Elastic_data = [Elastic_grid_MSE,Elastic_Kfold_MSE,Elastic_test_mse,Elastic_test_r2]

    return Elastic_data

def Random_Forest(X_train, X_test, y_train, y_test):
    def find_best_params(X_train,y_train):
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()
    
        param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']}

        grid_search = GridSearchCV(estimator=model,param_grid=param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
        grid_search.fit(X_train,y_train)
        RF_grid_MSE =-grid_search.best_score_
        print("RF: Best parameters found: ", grid_search.best_params_)
        print("RF: Best CV MSE: ", RF_grid_MSE)

        best_model = grid_search.best_estimator_
        kfold_scores = cross_val_score(best_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
        RF_Kfold_MSE =-kfold_scores.mean()
        print("RF: Best model cross-validated MSE: {:.4f}".format(RF_Kfold_MSE))

        return best_model,RF_grid_MSE,RF_Kfold_MSE
    
    def evaluate_on_test_set(best_model,X_test,y_test):
        y_pred = best_model.predict(X_test)

        RF_test_mse = mean_squared_error(y_test, y_pred)
        RF_test_r2 = r2_score(y_test, y_pred)

        print(f"RF: Test set MSE: {RF_test_mse}")
        print(f"RF: Test set R^2: {RF_test_r2}")

        return RF_test_mse,RF_test_r2

    best_model,RF_grid_MSE,RF_Kfold_MSE = find_best_params(X_train,y_train)
    RF_test_mse,RF_test_r2 = evaluate_on_test_set(best_model,X_test,y_test)

    RF_data = [RF_grid_MSE,RF_Kfold_MSE,RF_test_mse,RF_test_r2]

    return RF_data

def XGBoost(X_train, X_test, y_train, y_test):
    def find_best_params(X_train,y_train):
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

        return best_model,XGB_grid_MSE,XGB_Kfold_MSE
    
    def evaluate_on_test_set(best_model,X_test,y_test):
        y_pred = best_model.predict(X_test)

        XGB_test_mse = mean_squared_error(y_test, y_pred)
        XGB_test_r2 = r2_score(y_test, y_pred)

        print(f"XGB: Test set MSE: {XGB_test_mse}")
        print(f"XGB: Test set R^2: {XGB_test_r2}")

        return XGB_test_mse,XGB_test_r2
    
    best_model,XGB_grid_MSE,XGB_Kfold_MSE = find_best_params(X_train,y_train)
    XGB_test_mse,XGB_test_r2 = evaluate_on_test_set(best_model,X_test,y_test)

    XGB_data = [XGB_grid_MSE,XGB_Kfold_MSE,XGB_test_mse,XGB_test_r2]

    return XGB_data

def SVR(X_train, X_test, y_train, y_test):
    def find_best_params(X_train,y_train):
        from sklearn.svm import SVR
        model = SVR()
    
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.1, 0.2, 0.5, 1],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

        grid_search = GridSearchCV(estimator=model,param_grid=param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
        grid_search.fit(X_train,y_train)
        SVR_grid_MSE =-grid_search.best_score_
        print("SVR: Best parameters found: ", grid_search.best_params_)
        print("SVR: Best CV MSE: ", SVR_grid_MSE)

        best_model = grid_search.best_estimator_
        kfold_scores = cross_val_score(best_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
        SVR_Kfold_MSE =-kfold_scores.mean()
        print("SVR: Best model cross-validated MSE: {:.4f}".format(SVR_Kfold_MSE))

        return best_model,SVR_grid_MSE,SVR_Kfold_MSE
    
    def evaluate_on_test_set(best_model,X_test,y_test):
        y_pred = best_model.predict(X_test)

        SVR_test_mse = mean_squared_error(y_test, y_pred)
        SVR_test_r2 = r2_score(y_test, y_pred)

        print(f"SVR: Test set MSE: {SVR_test_mse}")
        print(f"SVR: Test set R^2: {SVR_test_r2}")

        return SVR_test_mse,SVR_test_r2
    
    best_model,SVR_grid_MSE,SVR_Kfold_MSE = find_best_params(X_train,y_train)
    SVR_test_mse,SVR_test_r2 = evaluate_on_test_set(best_model,X_test,y_test)

    SVR_data = [SVR_grid_MSE,SVR_Kfold_MSE,SVR_test_mse,SVR_test_r2]  
    return SVR_data  

def Polynomial_Regression():
    def Poly_preprocessing():
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LinearRegression

        data = pd.read_csv('crop_yield.csv')
        X = data.drop('Production', axis=1)
        y = data['Production']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        OneHotColumns = ['Season', 'State']
        encoder = OneHotEncoder(sparse_output=False, drop='first')

        X_train_onehot = encoder.fit_transform(X_train[OneHotColumns])
        X_test_onehot = encoder.transform(X_test[OneHotColumns])

        X_train_onehot_df = pd.DataFrame(X_train_onehot, columns=encoder.get_feature_names_out(OneHotColumns))
        X_test_onehot_df = pd.DataFrame(X_test_onehot, columns=encoder.get_feature_names_out(OneHotColumns))

        X_train = pd.concat([X_train.drop(OneHotColumns + ['Yield'], axis=1).reset_index(drop=True), X_train_onehot_df.reset_index(drop=True)], axis=1)
        X_test = pd.concat([X_test.drop(OneHotColumns + ['Yield'], axis=1).reset_index(drop=True), X_test_onehot_df.reset_index(drop=True)], axis=1)
    
        sc = StandardScaler()
        scColumns = ['Crop_Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

        X_train[scColumns] = sc.fit_transform(X_train[scColumns])
        X_test[scColumns] = sc.transform(X_test[scColumns])

        pipeline =Pipeline([
            ('poly',PolynomialFeatures()),
            ('linear', LinearRegression())])

        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train[scColumns])
        X_test_poly = poly.transform(X_test[scColumns])

        X_train_poly_df = pd.DataFrame(X_train_poly,columns=poly.get_feature_names_out(scColumns))
        X_test_poly_df = pd.DataFrame(X_test_poly,columns= poly.get_feature_names_out(scColumns))

        X_train = pd.concat([X_train.reset_index(drop=True), X_train_poly_df.reset_index(drop=True)],axis=1)
        X_test = pd.concat([X_test.reset_index(drop=True),X_test_poly_df.reset_index(drop=True)],axis=1)

        return X_train,X_test,y_train,y_test,pipeline
    
    def find_best_params(X_train,y_train,pipeline):
        param_grid = {
            'poly__degree':[2,3,4],
            'poly__interaction_only':[False,True],
            'linear__fit_intercept':[True,False]}

        grid_search = GridSearchCV(estimator=pipeline,param_grid=param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
        grid_search.fit(X_train,y_train)
        Poly_grid_MSE =-grid_search.best_score_
        print("Poly: Best parameters found: ", grid_search.best_params_)
        print("Poly: Best CV MSE: ", Poly_grid_MSE)

        best_model = grid_search.best_estimator_
        kfold_scores = cross_val_score(best_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
        Poly_Kfold_MSE =-kfold_scores.mean()
        print("Poly: Best model cross-validated MSE: {:.4f}".format(Poly_Kfold_MSE))

        return best_model,Poly_grid_MSE,Poly_Kfold_MSE
    
    def evaluate_on_test_set(best_model,X_test,y_test):
        y_pred = best_model.predict(X_test)

        Poly_test_mse = mean_squared_error(y_test, y_pred)
        Poly_test_r2 = r2_score(y_test, y_pred)

        print(f"Poly: Test set MSE: {Poly_test_mse}")
        print(f"Poly: Test set R^2: {Poly_test_r2}")

        return Poly_test_mse,Poly_test_r2

    X_train,X_test,y_train,y_test,pipeline = Poly_preprocessing()
    best_model,Poly_grid_MSE,Poly_Kfold_MSE = find_best_params(X_train,y_train,pipeline)
    Poly_test_mse,Poly_test_r2 = evaluate_on_test_set(best_model,X_test,y_test)

    Poly_data = [Poly_grid_MSE,Poly_Kfold_MSE,Poly_test_mse,Poly_test_r2]

    return Poly_data

def comparison(Elastic_data,RF_data, XGB_data, SVR_data, Poly_data):
    metrics = ['GridSearchCV MSE', 'KFold MSE', 'Test Set MSE', 'Test Set R2']
    comparison_df = pd.DataFrame({
        'Metrics': metrics,
        'ElasticNet': Elastic_data,
        'Random Forest': RF_data,
        'XGBoost': XGB_data,
        'SVR': SVR_data,
        'Polynomial Regression': Poly_data
    })
    print(comparison_df)
    comparison_df.to_csv('model_comparison.csv', index=False)

    return comparison_df 

X_train, X_test, y_train, y_test = preprocessing()
Elastic_data = ElasticNet(X_train, X_test, y_train, y_test)
RF_data = Random_Forest(X_train, X_test, y_train, y_test)
XGB_data = XGBoost(X_train, X_test, y_train, y_test)
SVR_data = SVR(X_train, X_test, y_train, y_test)
Poly_data = Polynomial_Regression()
comparison_df = comparison(Elastic_data,RF_data, XGB_data, SVR_data, Poly_data)