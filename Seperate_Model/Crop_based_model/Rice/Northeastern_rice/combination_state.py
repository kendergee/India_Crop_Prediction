import pandas as pd
import os
folder_path ='/Users/kendergee/Desktop/vscode/India_Crop_Prediction/Seperate_Model/Crop_based_model/Rice/Northeastern_rice'

combined_df = pd.DataFrame()
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
        df = pd.read_csv(file_path, encoding='utf-8',index_col=0)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        continue
    combined_df = pd.concat([combined_df, df], ignore_index=True)

combined_df.to_csv('/Users/kendergee/Desktop/vscode/India_Crop_Prediction/Seperate_Model/Crop_based_model/Rice/Northeastern_rice/Northeastern_rice.csv',index=False)
print(len(combined_df))