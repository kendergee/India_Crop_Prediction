import pandas as pd
data = pd.read_csv('/Users/kendergee/Desktop/vscode/India_Crop_Prediction/Seperate_CSV_origin/crop_yield.csv')
season_unique = data['Season'].unique()
print(season_unique)