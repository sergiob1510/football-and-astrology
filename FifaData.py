import numpy as np
import pandas as pd
import sqlite3, os.path
from sklearn import preprocessing
from helpers import zodiac_sign, position_replacer, cramers_V, cramers_application, heatmap_builder


# Retrieving Data
db = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.sqlite'))
df_attributes = pd.read_sql_query('SELECT * FROM Player_Attributes', db)
df_players = pd.read_sql_query('SELECT * FROM Player', db)
df_positions = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CompleteDataset.csv'))

# Data work

# Merging Dataframes
df = pd.merge(df_players, df_attributes, how='inner', on=['player_fifa_api_id'], copy=False)
df = df.merge(df_positions[['player_fifa_api_id', 'Preferred Positions']], how='inner', on=['player_fifa_api_id'], copy=False)

# Selecting useful columns
columns_of_interest = ['player_name', 'birthday', 'preferred_foot', 'Preferred Positions']
df = df[columns_of_interest]

#Cleaning duplicates and empty values
(rows_before, cols_before) = df.shape
print(f'There are {rows_before} records in the initial dataframe')
df = df.drop_duplicates(subset='player_name', keep='last')
df = df.dropna()
(rows_after, cols_after) = df.shape
print(f'There are {rows_after} records in the cleaned dataframe')
print(f'A total of {rows_before - rows_after} records were eliminated')

#Adding Zodiacal sign
df['birthday'] = pd.to_datetime(df['birthday'])
df['zodiacal_sign'] = df['birthday'].apply(lambda x: zodiac_sign(x.day, x.month))

#Changing Preferred Positions to something useful
df['Preferred Positions'] = df['Preferred Positions'].str.split()
df['camp_position'] = df['Preferred Positions'].apply(lambda x: position_replacer(x))


# Data Manipulation and Analysis

#Calculating relationship between variables
label = preprocessing.LabelEncoder()
data_encoded = pd.DataFrame()
for i in df[['preferred_foot', 'camp_position', 'zodiacal_sign']]:
    data_encoded[i]=label.fit_transform(df[i])

df2 = pd.DataFrame(cramers_application(data_encoded), columns=data_encoded.columns, index=data_encoded.columns)

#Heat Map creation
heatmap_builder(df2)
