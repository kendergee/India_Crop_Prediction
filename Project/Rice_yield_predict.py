import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import os
import json

current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# 獲取配置中的路徑
Model_directory = config['model_directory']

data_directory = config['data_directory']
regions = config['regions']

region = ['Central','East','North','Northeast','South','West']

seasons = ['Whole Year','Kharif','Rabi','Autumn','Summer','Winter']

states_dict = {
    'Central':['Chhattisgarh','Madhya Pradesh'],
    'East':['Bihar','Jharkhand'],
    'North':['Delhi','Haryana','Himachal Pradesh','Jammu and Kashmir','Punjab','Uttar Pradesh','Uttarakhand'],
    'Northeast':['Arunachal Pradesh','Assam','Manipur','Meghalaya','Mizoram','Nagaland','Odisha','Sikkim','Tripura','West Bengal'],
    'South':['Andhra Pradesh','Karnataka','Kerala','Puducherry','Tamil Nadu','Telangana'],
    'West':['Goa','Gujarat','Maharashtra']
}

resources = {}

random_state_paramater ={
    'Central' : 0,
    'East':0,
    'North':0,
    'Northeast':42,
    'South':42,
    'West':42
}

def userinput():
    print('歡迎使用印度稻米預測器')
    print('請問你想選擇哪個區域')
    for i, rg in enumerate(region):
        print(f'{i+1}:{rg}')
    region_index = int(input('請選擇一個區域（輸入對應的數字）：'))-1
    selected_region = region[region_index]
    print(f'你選擇的區域是：{selected_region}')

    states = states_dict[selected_region]
    print('可選擇的州有以下：')
    for i, state in enumerate(states):
        print(f'{i+1}:{state}')
    state_index = int(input('請選擇一個州（輸入對應的數字）:'))-1
    selected_state = states[state_index]
    print(f'你選擇的州是：{selected_state}')
    
    print('可選擇的季節有以下：')
    for i, season in enumerate(seasons):
        print(f'{i+1}:{season}')
    season_index = int(input('請選擇一個季節（輸入對應的數字）：'))-1
    selected_season = seasons[season_index]
    print(f'你選擇的季節是：{selected_season}')

    selected_crop_years = int(input('請輸入西元年份：'))
    selected_area = float(input('請輸入種植面積：'))
    selected_production = float(input('請輸入作物產量：'))
    selected_annual_rainfall = float(input('請輸入年降雨量：'))
    selected_fertilizer = float(input('請輸入肥料用量：'))
    selected_pesticide = float(input('請輸入害蟲總量：'))

    ans_dict ={
        'ans_categorical' : [selected_season,selected_state],
        'ans_numeric':[selected_crop_years,selected_area,selected_production,selected_annual_rainfall,selected_fertilizer,selected_pesticide]
    }


    return selected_region,ans_dict

def load_resources(selected_region):
    if selected_region not in resources:
        Model_path = os.path.join(Model_directory,f'{selected_region}_model.pkl')
        data_path = os.path.join(data_directory,f'{selected_region}_rice.csv')

        Model = joblib.load(Model_path)
        data = pd.read_csv(data_path)
    
    print('資料加載完成')
    return Model,data

def work(Model,data,ans_dict,selected_region):
    def preprocessing(data,ans_dict,selected_region):
        data = data.dropna()
        X = data.drop(['Yield'], axis=1)
        y = data['Yield']
        
        random_state_input = random_state_paramater[selected_region]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state_input)

        OneHotColumns = ['Season', 'State']
        encoder = OneHotEncoder(sparse_output=False, drop='first',handle_unknown='ignore')

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

        ans_categorical = ans_dict['ans_categorical']
        ans_numeric = ans_dict['ans_numeric']
        ans_categorical = pd.DataFrame([ans_categorical],encoder.get_feature_names_out(OneHotColumns))
        ans_numeric = pd.DataFrame([ans_numeric])
        ans_categorical = encoder.transform(ans_categorical)
        ans_numeric = sc.transform(ans_numeric)
        ans_categorical = pd.DataFrame(ans_categorical)
        ans_numeric = pd.DataFrame(ans_numeric,columns=['Crop_Year','Area','Production','Annual_Rainfall','Fertilizer','Pesticide'])
        ans_completion = pd.concat([ans_numeric,ans_categorical],axis=1)
    
        missing_cols = set(X_train.columns) - set(ans_completion.columns)
        for col in missing_cols:
            ans_completion[col] = 0
        ans_completion = ans_completion[X_train.columns]        
        
        print('資料前處理完成')
        
        return X_train, X_test, y_train, y_test,ans_completion
    
    def prediction(Model,X_train,X_test,y_train,y_test,ans_completion):
        model = Model.fit(X_train,y_train)
        y_pred = model.predict(ans_completion)
        print('預測單位產量如下：')
        print(y_pred[0])
    
    X_train, X_test, y_train, y_test,ans_completion = preprocessing(data,ans_dict,selected_region)
    prediction(Model,X_train,X_test,y_train,y_test,ans_completion)

selected_region,ans_dict = userinput()
Model,data = load_resources(selected_region)
work(Model,data,ans_dict,selected_region)