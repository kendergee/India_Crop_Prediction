import pandas as pd
import os

folder_path ='/Users/kendergee/Desktop/vscode/India_Crop_Prediction/Seperate_Model/Crop_based_model/Rice/Rice_states_data'
file_counts = {}

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path,filename)
    df = pd.read_csv(file_path)
    file_counts[filename] = len(df)

sorted_file_counts = dict(sorted(file_counts.items(), key=lambda item: item[1], reverse=True))

for file,counts in sorted_file_counts.items():
    print(f'{file}:{counts}筆資料')