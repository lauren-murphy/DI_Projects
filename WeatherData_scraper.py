
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import date, timedelta
#get webpage with 2016 weather data
page = requests.get('https://www.wunderground.com/history/airport/KATL/2016/1/1/CustomHistory.html?dayend=31&monthend=12&yearend=2016&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=')
#create BS object
soup = BeautifulSoup(page.text, 'html.parser')

table = soup.find(id='obsTable')
values = table.find_all(class_='wx-value')
column_names = ['date','temp_high','temp_avg','temp_low','dew_high','dew_avg','dew_low','humidity_high','humidity_avg','humidity_low',
                'pressure_high',"pressure_avg","pressure_low",'visibility_high','visibility_avg','visibility_low',
                'wind_high',"wind_avg","wind_high2",'precipitation']
tally = 0
cal_date = date(2016,1,1)
value_dict = {}

for col in column_names:
    value_dict[col] = []


for val in values:
    column_number = tally % len(column_names)
    col_name = column_names[column_number]
    if column_number == 0:
        date_string = cal_date.strftime('%d-%b-%Y')
        value_dict[col_name].append(date_string)
        tally += 1
        column_number = tally % len(column_names)
        col_name = column_names[column_number]
        value_dict[col_name].append(float(val.next))
    elif col_name == 'wind_high2' and (cal_date == date(2016, 2, 6) or cal_date == date(2016, 9, 17)):
        value_dict[col_name].append(float(0))
        tally += 1
        column_number = tally % len(column_names)
        col_name = column_names[column_number]
        value_dict[col_name].append(float(val.next))
    elif val.next == 'T':
        value_dict[col_name].append(float(0))
    else:
        value_dict[col_name].append(float(val.next))
    tally += 1
    if col_name == column_names[-1]:
        cal_date = cal_date + timedelta(days=1)

df = pd.DataFrame(data=value_dict)
df.to_csv('weatherData_2016.csv')

print('Hello!')