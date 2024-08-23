import pandas as pd
import os

data = pd.read_csv('crop_yield.csv')
Crops = data['Crop'].unique()

crop_dataframes ={}
save_directory ='/Users/kendergee/Desktop/vscode/Indian_Crop_Prediction/Indian_Crop_Prediction/Seperate_CSV/Many_Crops'

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

for crop in Crops:
    crop_df = data[data['Crop']== crop].reset_index(drop=True)
    crop_dataframes[crop] = crop_df

    save_path = os.path.join(save_directory,f'{crop}_data.csv')

    parent_directory = os.path.dirname(save_path)
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)

    crop_df.to_csv(save_path,index=False)
    print(f'Saved {crop} data to {save_path}')