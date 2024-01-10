import numpy as np          #   used to make arrays
import pandas as pd         #   used to make dataframes / structured tables for the data   
import matplotlib.pyplot as plt         #   used to create plots   
import seaborn as sns       #   used to create graphs
from sklearn.model_selection import train_test_split        #   used to split our data into training data and test data
from xgboost import XGBRegressor        #   provides the gradient boosting algorithm
from sklearn import metrics         #   used to evaluate our model
import os

###     DATA COLLECTION AND PROCESSING

#   loading data from CSV to pd DataFrame
cwd = os.getcwd()       #   to import data from relative path
calories = pd.read_csv(f'{cwd}/dataset/calories.csv')
exercise_data = pd.read_csv(f'{cwd}/dataset/exercise.csv')

#   printing first 5 rows of the DFs
#print(calories.head())
''' Output:
    User_ID  Calories
0  14733363     231.0
1  14861698      66.0
2  11179863      26.0
3  16180408      71.0
4  17771927      35.0
'''

#print(exercise_data.head())
''' Output:
    User_ID  Gender  Age  Height  Weight  Duration  Heart_Rate  Body_Temp
0  14733363    male   68   190.0    94.0      29.0       105.0       40.8
1  14861698  female   20   166.0    60.0      14.0        94.0       40.3
2  11179863    male   69   179.0    79.0       5.0        88.0       38.7
3  16180408  female   34   179.0    71.0      13.0       100.0       40.5
4  17771927  female   27   154.0    58.0      10.0        81.0       39.8
'''

###     COMBINING THE DFs

calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)        #   data appended after last column
#print(calories_data.head())
''' Output:
    User_ID  Gender  Age  ...  Heart_Rate  Body_Temp  Calories
0  14733363    male   68  ...       105.0       40.8     231.0
1  14861698  female   20  ...        94.0       40.3      66.0
2  11179863    male   69  ...        88.0       38.7      26.0
3  16180408  female   34  ...       100.0       40.5      71.0
4  17771927  female   27  ...        81.0       39.8      35.0
'''

#print(calories_data.shape)            # (# data pts, # columns)
''' Output:
(15000, 9)
'''

# print(calories_data.info())           # insight on data
''' Output:
RangeIndex: 15000 entries, 0 to 14999
Data columns (total 9 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   User_ID     15000 non-null  int64  
 1   Gender      15000 non-null  object 
 2   Age         15000 non-null  int64  
 3   Height      15000 non-null  float64
 4   Weight      15000 non-null  float64
 5   Duration    15000 non-null  float64
 6   Heart_Rate  15000 non-null  float64
 7   Body_Temp   15000 non-null  float64
 8   Calories    15000 non-null  float64
dtypes: float64(6), int64(2), object(1)
memory usage: 1.0+ MB
'''

#print(calories_data.isnull().sum())     #   check if there are any missing values in the columns
''' Output:
User_ID       0
Gender        0
Age           0
Height        0
Weight        0
Duration      0
Heart_Rate    0
Body_Temp     0
Calories      0
dtype: int64
'''

###     DATA ANALYSIS
