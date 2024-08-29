import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import  matplotlib.pyplot as plt
import os
import json
import warnings 
import tkinter as tk
from tkinter import ttk

warnings.filterwarnings('ignore', category=UserWarning)

current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

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

def update_states(*args):
    selected_region = region_var.get()
    state_menu['values'] = states_dict.get(selected_region, [])

def transfer():
    # 获取输入值
    selected_region = region_var.get()
    selected_state = state_var.get()
    selected_season = season_var.get()
    selected_crop_years = year_var.get()
    selected_area = area_var.get()
    selected_annual_rainfall = rainfall_var.get()
    selected_fertilizer = fertilizer_var.get()
    selected_pesticide = pesticide_var.get()

    ans_dict ={
        'ans_categorical' : [selected_season, selected_state],
        'ans_numeric': [selected_crop_years, selected_area, selected_annual_rainfall, selected_fertilizer, selected_pesticide]
    }

    return selected_region, ans_dict

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
        X = data.drop(['Yield','Production'], axis=1)
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
        scColumns = ['Crop_Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

        X_train[scColumns] = sc.fit_transform(X_train[scColumns])
        X_test[scColumns] = sc.transform(X_test[scColumns])

        ans_categorical = ans_dict['ans_categorical']
        ans_numeric = ans_dict['ans_numeric']
        ans_categorical = pd.DataFrame([ans_categorical],encoder.get_feature_names_out(OneHotColumns))
        ans_numeric = pd.DataFrame([ans_numeric])
        ans_categorical = encoder.transform(ans_categorical)
        ans_numeric = sc.transform(ans_numeric)
        ans_categorical = pd.DataFrame(ans_categorical)
        ans_numeric = pd.DataFrame(ans_numeric,columns=['Crop_Year','Area','Annual_Rainfall','Fertilizer','Pesticide'])
        ans_completion = pd.concat([ans_numeric,ans_categorical],axis=1)
    
        missing_cols = set(X_train.columns) - set(ans_completion.columns)
        for col in missing_cols:
            ans_completion[col] = 0
        ans_completion = ans_completion[X_train.columns]
        ans_completion = ans_completion[:1]        
        
        print('資料前處理完成')
        
        return X_train, X_test, y_train, y_test,ans_completion
    
    def prediction(Model,X_train,X_test,y_train,y_test,ans_completion):
        model = Model.fit(X_train,y_train)
        y_pred = model.predict(ans_completion)
        y_pred = round(y_pred[0],3)
        print('預測單位產量如下：')
        print(f'每公頃約{y_pred}公噸的單位產量')
        return y_pred 
    
    X_train, X_test, y_train, y_test,ans_completion = preprocessing(data,ans_dict,selected_region)
    y_pred = prediction(Model,X_train,X_test,y_train,y_test,ans_completion)
    return y_pred

def plot_yield_data(data,y_pred,ans_dict):
    state_data = data[data['State'] == ans_dict['ans_categorical'][1]]
    plt.figure(figsize=(10,6))
    plt.scatter(state_data['Crop_Year'],state_data['Yield'],label='Actual Yield')
    plt.scatter(ans_dict['ans_numeric'][0],y_pred,label='Predicted Yield',color='red')
    plt.title(f"{ans_dict['ans_categorical'][1]} past yield data")
    plt.xlabel("Year")
    plt.ylabel("Yield(tonnes/ha)")
    plt.legend()
    plt.grid(True)
    plt.show()


def project_process():
    # 獲取輸入資料
    selected_region, ans_dict = transfer()
    
    # 加載模型和資料
    Model, data = load_resources(selected_region)
    
    # 執行預測
    y_pred =work(Model, data, ans_dict, selected_region)

    result_label.config(text=f'每公頃約 {y_pred} 公噸的單位產量')

    plot_yield_data(data,y_pred,ans_dict)



root = tk.Tk()
root.title("印度稻米預測器")

# 設定ttk風格
style = ttk.Style()
style.theme_use("clam")

# 第一個輸入區域：選擇區域
region_label = ttk.Label(root, text="請選擇區域：")
region_label.grid(row=0, column=0, padx=10, pady=5, sticky="W")

region_var = tk.StringVar()
region_menu = ttk.Combobox(root, textvariable=region_var)
region_menu['values'] = region
region_menu.grid(row=0, column=1, padx=10, pady=5)
region_menu.bind('<<ComboboxSelected>>', update_states)

# 第二個輸入區域：選擇邦
state_label = ttk.Label(root, text="請選擇邦：")
state_label.grid(row=1, column=0, padx=10, pady=5, sticky="W")

state_var = tk.StringVar()
state_menu = ttk.Combobox(root, textvariable=state_var)
state_menu.grid(row=1, column=1, padx=10, pady=5)

# 第三個輸入區域：選擇季節
season_label = ttk.Label(root, text="請選擇季節：")
season_label.grid(row=2, column=0, padx=10, pady=5, sticky="W")

season_var = tk.StringVar()
season_menu = ttk.Combobox(root, textvariable=season_var)
season_menu['values'] = seasons
season_menu.grid(row=2, column=1, padx=10, pady=5)

# 其他輸入區域：年份、面積、雨量、肥料用量、農藥用量
year_label = ttk.Label(root, text="年份：")
year_label.grid(row=3, column=0, padx=10, pady=5, sticky="W")

year_var = tk.IntVar()
year_entry = ttk.Entry(root, textvariable=year_var)
year_entry.grid(row=3, column=1, padx=10, pady=5)

area_label = ttk.Label(root, text="面積 (公頃)：")
area_label.grid(row=4, column=0, padx=10, pady=5, sticky="W")

area_var = tk.DoubleVar()
area_entry = ttk.Entry(root, textvariable=area_var)
area_entry.grid(row=4, column=1, padx=10, pady=5)

rainfall_label = ttk.Label(root, text="雨量 (毫米)：")
rainfall_label.grid(row=5, column=0, padx=10, pady=5, sticky="W")

rainfall_var = tk.DoubleVar()
rainfall_entry = ttk.Entry(root, textvariable=rainfall_var)
rainfall_entry.grid(row=5, column=1, padx=10, pady=5)

fertilizer_label = ttk.Label(root, text="肥料用量 (公斤)：")
fertilizer_label.grid(row=6, column=0, padx=10, pady=5, sticky="W")

fertilizer_var = tk.DoubleVar()
fertilizer_entry = ttk.Entry(root, textvariable=fertilizer_var)
fertilizer_entry.grid(row=6, column=1, padx=10, pady=5)

pesticide_label = ttk.Label(root, text="農藥用量 (公升)：")
pesticide_label.grid(row=7, column=0, padx=10, pady=5, sticky="W")

pesticide_var = tk.DoubleVar()
pesticide_entry = ttk.Entry(root, textvariable=pesticide_var)
pesticide_entry.grid(row=7, column=1, padx=10, pady=5)

predict_button = ttk.Button(root, text="預測", command=project_process)
predict_button.grid(row=8, column=0, columnspan=2, pady=10)

result_label = ttk.Label(root, text='')
result_label.grid(row=9, column=0, columnspan=2, pady=10)

# 啟動主事件循環
root.mainloop()