import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

df_a = pd.read_csv("./Stats19_Data_2005-2014/Accidents0514.csv")
#df_c = pd.read_csv("./Stats19_Data_2005-2014/Casualties0514.csv")
#df_v = pd.read_csv("./Stats19_Data_2005-2014/Vehicles0514.csv")

df_a.head()

##need to find out if 1 == Urban or 2 == Urban

isUrban = 0
for x in df_a['Urban_or_Rural_Area']:
    if x == 1:
        isUrban += 1

result = isUrban / len(df_a['Urban_or_Rural_Area'])
print(result)

date = df_a.loc[:,'Date']
day_list = []
month_list = []
year_list = []

for x in range(len(df_a)): #separate out date data into day month year, save the year data in a year_list
    day,month,year = date[x].split('/')
    day_list.append(int(day))
    month_list.append(int(month))
    year_list.append(int(year))

df_a['Year'] = year_list
year_bins = df_a.Year.unique()

sns.distplot(df_a['Year'], kde=False, rug=True)
slope, intercept, r_value, p_value, std_err = stats.linregress(df_a['Year'],df_a['Accident_Index'])
print(slope)