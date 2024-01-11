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

#   get some statistical measures about the data
#print(calories_data.describe())
''' Output:
            User_ID           Age  ...     Body_Temp      Calories
count  1.500000e+04  15000.000000  ...  15000.000000  15000.000000
mean   1.497736e+07     42.789800  ...     40.025453     89.539533
std    2.872851e+06     16.980264  ...      0.779230     62.456978
min    1.000116e+07     20.000000  ...     37.100000      1.000000
25%    1.247419e+07     28.000000  ...     39.600000     35.000000
50%    1.499728e+07     39.000000  ...     40.200000     79.000000
75%    1.744928e+07     56.000000  ...     40.600000    138.000000
max    1.999965e+07     79.000000  ...     41.500000    314.000000

'''

###     DATA VISUALIZATION

sns.set()       #   gives a theme of our plot

#   plotting the gender column in count plot
#sns.countplot(calories_data['Gender'])
#plt.show()

#   finding the distribution of "Age"
#sns.distplot(calories_data['Age'])
#plt.show()

#   finding the distribution of "Height"
#sns.distplot(calories_data['Height'])
#plt.show()

#   finding the distribution of "Weight"
#sns.distplot(calories_data['Weight'])
#plt.show()

###     CONVERTING STR TO NUM VALUES (machine cannot interpret str)

calories_data.replace(
    {'Gender':
     {'male':0, 
      'female':1
      }
    }, 
    inplace=True)

###     FINDING THE CORRELATION IN THE DATASET (positive or negative)

correlation = calories_data.corr()

#   constructing a heatmap to interpret correlation
'''
plt.figure(figsize=(10,10))
sns.heatmap(
    correlation, 
    cbar=True, 
    square=True, 
    fmt='.1f', 
    annot=True, 
    annot_kws={'size':8},   
    cmap='Blues'  
    )
plt.show()
'''

###   SEPARATING FEATURES AND TARGET

x = calories_data.drop(columns=['User_ID', 'Calories'], axis=1)     #   these columns are of no use in this analysis
y = calories_data['Calories']       #   store this specific column 

#print(x)
''' Output:
       Gender  Age  Height  Weight  Duration  Heart_Rate  Body_Temp
0           0   68   190.0    94.0      29.0       105.0       40.8
1           1   20   166.0    60.0      14.0        94.0       40.3
2           0   69   179.0    79.0       5.0        88.0       38.7
3           1   34   179.0    71.0      13.0       100.0       40.5
4           1   27   154.0    58.0      10.0        81.0       39.8
...       ...  ...     ...     ...       ...         ...        ...
14995       1   20   193.0    86.0      11.0        92.0       40.4
14996       1   27   165.0    65.0       6.0        85.0       39.2
14997       1   43   159.0    58.0      16.0        90.0       40.1
14998       0   78   193.0    97.0       2.0        84.0       38.3
14999       0   63   173.0    79.0      18.0        92.0       40.5
'''
#print(y)
''' Output:
0        231.0
1         66.0
2         26.0
3         71.0
4         35.0
         ...  
14995     45.0
14996     23.0
14997     75.0
14998     11.0
14999     98.0
'''

###     SPLITTING DATA INTO TRAINING AND TEST DATA

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=2)
#print(x.shape, x_train.shape, x_test.shape)
''' Output:
    (15000, 7) 
    (12000, 7) 
    (3000, 7)
'''

###     MODEL TRAINING  

#   loading model
model = XGBRegressor()

#   training model with x_train
model.fit(x_train, y_train) 

###     EVALUATION

#   prediction on test data
test_data_prediction = model.predict(x_test)
#print(test_data_prediction)
''' Output:
    [125.58828  222.11377   38.725952 ... 144.3179    23.425894  90.100494]
'''

#   mean absolute error
mae = metrics.mean_absolute_error(y_test, test_data_prediction)
#print(f"Mean Absolute Error := {mae}")
''' Output:
    Mean Absolute Error := 1.4833678883314132   #   very minuscule difference
'''
