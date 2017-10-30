#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:30:07 2017

@author: laurenmurphy
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.cluster.vq import kmeans2, whiten
sns.set_style('whitegrid')
from scipy import stats
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.stats import ttest_rel
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import date, timedelta

os.chdir('/Users/laurenmurphy/Documents/ComputerShit/DataIncubator/')

#initiallize data sets
df = pd.read_csv('avl_otpdata_year.csv')
#clean data
df = df.dropna(axis=0, how='any') #remove NANs
df.latitude = df.latitude * 0.0000001 #scale to actual lat values
df.longitude = df.longitude * 0.0000001 #scale to actual lat values #reverse to show delays as +time
df.scheduled_time = df.scheduled_time / 60 # scale time from seconds to minutes elapsed from start of service 

#initial exploration, identify the features common to late buses
#subset for only late buses 
late_bus_df = df[df.adherence_seconds < -600] #select only times that are > 10 minutes behind schedule

#coorelations between scheduled time and delay
r = np.corrcoef(list(late_bus_df.adherence_seconds),list(late_bus_df.scheduled_time))
print('correlation between bus delay and schedule time', r)
slope, intercept, r_value, p_value, std_err = stats.linregress(list(late_bus_df.scheduled_time),list(late_bus_df.adherence_seconds))
line = slope*list(late_bus_df.scheduled_time)+intercept
print('slope' , slope, 'intercept', intercept, 'r_value', r_value, 'p_value', p_value, 'std_err', std_err)


#examine the features of early buses 
early_bus_df = df[df.adherence_seconds >= 0]
#coorelations between scheduled time and delay
r = np.corrcoef(list(early_bus_df.adherence_seconds),list(early_bus_df.scheduled_time))
print('correlation between bus ahead of schedule and schedule time', r)
slope, intercept, r_value, p_value, std_err = stats.linregress(list(early_bus_df.scheduled_time),list(early_bus_df.adherence_seconds))
print('slope' , slope, 'intercept', intercept, 'r_value', r_value, 'p_value', p_value, 'std_err', std_err)

#create subplot with late and early bus data
f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
plt.grid(True)
ax1.set_title('Time Peaks of Late MARTA Busses')
ax1.scatter(list(late_bus_df.scheduled_time),list(late_bus_df.adherence_seconds * -1), facecolor='r', alpha = 0.005)
ax1.set_ylabel('Behind Schedule (s)')
ax1.set_yticks([600, 800, 1000, 1200])
ax2.scatter(list(early_bus_df.scheduled_time),list(early_bus_df.adherence_seconds * -1), facecolor='b', alpha = 0.005)
plt.xlabel('Time of Day')
ax2.set_ylabel('Ahead of Schedule (s)')
ax2.set_xticklabels(['1:00','4:00', '7:00', '10:00','13:00','16:00','19:00','21:00', '24:00'])
ax2.set_yticks([0, -200, -400, -600])
ax2.set_ylim([-600, 0])
plt.savefig("MARTADelaysByTime_subplot.png", dpi=120)
plt.show()

#Next: examine only buses during the evening rush hour 

##subset for times between 60000 and 70000 as we IDd these as being the most delayed
evening_bus = df[df.scheduled_time > (60000/60)][df.scheduled_time < (70000/60)] #select the time window
evening_bus_by_route = evening_bus.groupby('route_abbr').mean() #group by route # to reduce dimenstionality
evening_coordinates = evening_bus_by_route.as_matrix(columns=['latitude','longitude']) #create an array of coordinate pairs
#evaluete means for visualization 
evening_mean_lat = evening_bus_by_route.latitude.mean()
evening_mean_long = evening_bus_by_route.longitude.mean()

#preliminary visualization of data scatter
plt.plot()
plt.title('Visualize Data')
plt.scatter(evening_coordinates[:,0], evening_coordinates[:,1])
plt.show()

# run K means to determine # of cluseters of late buses
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(evening_coordinates)
    kmeanModel.fit(evening_coordinates)
    distortions.append(sum(np.min(cdist(evening_coordinates, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / evening_coordinates.shape[0])
 
# Plot the elbow
plt.plot()
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.savefig('ElbowPlotOfLateEveningBuses.png')
plt.show()
#drop at two, slight drop at 4, may need more than 10 clusters! 

#run k means clustering with 2 clusters to determine geo clusters of MARTA activity 
x, y = kmeans2(whiten(evening_coordinates), 2, iter = 20)
plt.scatter(evening_coordinates[:,0], evening_coordinates[:,1], s=400, c=y, alpha = .5, cmap=plt.cm.autumn)
plt.grid(False)
plt.show()

#run k means clustering with 4 clusters to determine geo clusters of MARTA activity 
x, y = kmeans2(whiten(evening_coordinates), 4, iter = 20)
plt.scatter(evening_coordinates[:,0], evening_coordinates[:,1], s=400, c=y, alpha = .5, cmap=plt.cm.autumn)
plt.grid(False)
plt.show()
#4 clusters appear to be a better fit for our preliminary analyses

#run scatter plot to visualize areas of greatest delays
plt.scatter(evening_coordinates[:,0], evening_coordinates[:,1], s=evening_bus_by_route.adherence_seconds * -7, c=evening_bus_by_route.adherence_seconds, alpha = 0.2, cmap=plt.cm.YlOrRd)
plt.scatter(evening_mean_lat, evening_mean_long, marker='*', c='k')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Map of Evening Delays by Delay in Seconds')
plt.grid(False)
plt.savefig('ScatterPlotOfLatAndLongScaledByDelay.png')
plt.show()

#weather would be an interesting feature to consider - but the MARTA data does not provide this info. Let's go find it. 

#get webpage with 2016 weather data
page = requests.get('https://www.wunderground.com/history/airport/KATL/2016/1/1/CustomHistory.html?dayend=31&monthend=12&yearend=2016&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=')
#create BS object
soup = BeautifulSoup(page.text, 'html.parser')

#find the object that contains the table
table = soup.find(id='obsTable')
#find the tag that holds the values
values = table.find_all(class_='wx-value')
#set the column names for all data
column_names = ['date','temp_high','temp_avg','temp_low','dew_high','dew_avg','dew_low','humidity_high','humidity_avg','humidity_low',
                'pressure_high','pressure_avg','pressure_low','visibility_high','visibility_avg','visibility_low',
                'wind_high','wind_avg','wind_high2','precipitation'] #two columns were named wind high - perhaps one is gusts? 

tally = 0 #intialize counter
cal_date = date(2016,1,1) #set the start date - jan 1. 
value_dict = {} #initialize disctionary 

for col in column_names: #create key values for each column name
    value_dict[col] = [] 

for val in values: #iterate through all of the values
    column_number = tally % len(column_names) #count the values to determine which column they belong in
    col_name = column_names[column_number] #get that name
    if column_number == 0: #create a date object for the data column
        date_string = cal_date.strftime('%d-%b-%y') #format it to look like the MARTA data
        value_dict[col_name].append(date_string) 
        tally += 1 #increment because it isn't actually pulling a value from the list
        column_number = tally % len(column_names) #move on to the next one
        col_name = column_names[column_number]
        value_dict[col_name].append(float(val.next))
    elif col_name == 'wind_high2' and (cal_date == date(2016, 2, 6) or cal_date == date(2016, 9, 17)): #create catches for the random '-' in the data
        value_dict[col_name].append(float(0)) #assign dashes a zero value
        tally += 1
        column_number = tally % len(column_names)
        col_name = column_names[column_number]
        value_dict[col_name].append(float(val.next))
    elif val.next == 'T': #T = trace amounts of precipitation. Consider this to be 0. 
        value_dict[col_name].append(float(0))
    else:
        value_dict[col_name].append(float(val.next))
    tally += 1
    if col_name == column_names[-1]:
        cal_date = cal_date + timedelta(days=1) #increment the date 

weather_df = pd.DataFrame(data=value_dict)
#weather_df.to_csv('weatherData_2016.csv')

##add in scrubbed weather data  from AccuWeather.com
weather_date = []
for x in weather_df.date:
    x = x.replace('-20', '-')
    weather_date.append(x.upper())    
weather_df.date = weather_date 
weather_df = weather_df.set_index('date') 

#Group bus by date to add weather by date 
bus_data_by_date = df.groupby('calendar_day').mean()

#join data on date index     
merged_data = bus_data_by_date.join(weather_df)
#analyze relationship between delay and hot weather
plt.scatter(list(np.log(merged_data.adherence_seconds*-1)),list(np.log(merged_data.temp_high)), facecolor='g', alpha = 0.1)
plt.show()
slope, intercept, r_value, p_value, std_err = stats.linregress(list(np.log(merged_data.adherence_seconds*-1)),list(np.log(merged_data.temp_high)))
print('slope' , slope, 'intercept', intercept, 'r_value', r_value, 'p_value', p_value, 'std_err', std_err)
r = np.corrcoef(list(np.log(merged_data.adherence_seconds*-1)),list(np.log(merged_data.temp_high)))
print('Heat and delay time ', r)
#analyze relationship between delay and cold weather
plt.scatter(list(np.log(merged_data.adherence_seconds*-1)),list(np.log(merged_data.temp_low)), facecolor='g', alpha = 0.1)
plt.show()
slope, intercept, r_value, p_value, std_err = stats.linregress(list(np.log(merged_data.adherence_seconds*-1)),list(np.log(merged_data.temp_low)))
print('slope' , slope, 'intercept', intercept, 'r_value', r_value, 'p_value', p_value, 'std_err', std_err)
r = np.corrcoef(list(np.log(merged_data.adherence_seconds*-1)),list(np.log(merged_data.temp_low)))
print('Cold and delay time ', r)
#analyze relationship between rain and cold weather
plt.scatter(list(np.log(merged_data.adherence_seconds*-1)),list(np.log(merged_data.precipitation)), facecolor='g', alpha = 0.1)
plt.show()
slope, intercept, r_value, p_value, std_err = stats.linregress(list(np.log(merged_data.adherence_seconds*-1)),list(np.log(merged_data.precipitation)))
print('slope' , slope, 'intercept', intercept, 'r_value', r_value, 'p_value', p_value, 'std_err', std_err)
r = np.corrcoef(list(np.log(merged_data.adherence_seconds*-1)),list(np.log(merged_data.precipitation)))
print('Rain and delay time ', r)
#separate into "rainy days" vs "nonrainy days" 
rainy_days = merged_data[merged_data.precipitation > 0]
non_rainy_days = merged_data[merged_data.precipitation < 0]
#analyze relationship between rain days and cold weather
plt.scatter(list(rainy_days.adherence_seconds),list(rainy_days.precip_y_n), facecolor='g', alpha = 0.25, s = 200)
plt.show()
slope, intercept, r_value, p_value, std_err = stats.linregress(list(rainy_days.adherence_seconds),list(rainy_days.precipitation))
print('slope' , slope, 'intercept', intercept, 'r_value', r_value, 'p_value', p_value, 'std_err', std_err)
r = np.corrcoef(list(rainy_days.adherence_seconds),list(rainy_days.precipitation))
print('Rainy days and delay time', r)
#create categorical variable for presence or absence of rain
precip_y_n = []
for x in merged_data.precipitation:
    if float(x) <= 0:
        precip_y_n.append(0) 
    else:
        precip_y_n.append(1) 
merged_data['precip_y_n'] = precip_y_n
#analyze this relationship
plt.scatter(list(merged_data.adherence_seconds),list(merged_data.precip_y_n), facecolor='g', alpha = 0.25, s = 200)
plt.show()
slope, intercept, r_value, p_value, std_err = stats.linregress(list(merged_data.adherence_seconds),list(merged_data.precip_y_n))
print('slope' , slope, 'intercept', intercept, 'r_value', r_value, 'p_value', p_value, 'std_err', std_err)
r = np.corrcoef(list(merged_data.adherence_seconds),list(merged_data.precip_y_n))
print('Rainy days and delay time', r)
#do a paired t-test to see if they are related
ttest_rel(merged_data.adherence_seconds,merged_data.precip_y_n)
#create a nice box plot with individual points overlaid to show distribution of delays on rainy and nonrainy days
sns.set_style('white')
ax = sns.boxplot(y='adherence_seconds', x='precip_y_n', data=merged_data, palette='BuGn_r')
ax = sns.swarmplot(y='adherence_seconds', x='precip_y_n', data=merged_data, color='k', alpha=0.5)
ax.set(xlabel='Presence of Precipitation', ylabel='Adherence Seconds', title="Average Daily Delay in Seconds on Clear and Rainy Days")
ax.set_xticklabels(['No precipitation','Precipitation'])
ax.invert_yaxis()
fig = ax.get_figure()
fig.savefig('precipitation_delays.png') 