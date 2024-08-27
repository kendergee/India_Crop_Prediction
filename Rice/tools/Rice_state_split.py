import pandas as pd
import os

data = pd.read_csv('/Users/kendergee/Desktop/vscode/India_Crop_Prediction/Seperate_Model/Crop_based_model/Rice/Rice_data.csv')
states = data['State'].unique()

state_dataframes ={}
save_directory ='/Users/kendergee/Desktop/vscode/India_Crop_Prediction/Seperate_Model/Crop_based_model/Rice/Rice_states_data'

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

for state in states:
    state_df = data[data['State']== state].reset_index(drop=True)
    state_dataframes[state] = state_df

    save_path = os.path.join(save_directory,f'{state}_data.csv')

    parent_directory = os.path.dirname(save_path)
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)

    state_df.to_csv(save_path,index=False)
    print(f'Saved {state} data to {save_path}')