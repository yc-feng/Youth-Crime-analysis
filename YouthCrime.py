#!/usr/bin/env python
# coding: utf-8

# # Youth Crime in Queensland

# ## Background and Introduction
# Youth crime is a pressing issue recently in Australia that directly affects the safety of communities. As the offences committed by juveniles happened frequently, people's concerns about the safety of life and property are increasing. In Figure 1, it shows the number of youth crime related news published by The Guardian in each month since 2021, it can be observed that the number is increasing since 2022 generally. 
# 
# As society pays more and more attention to the issue of youth crime, policymakers can play a significant role in solving this problem. The purpose of this research is to produce a comprehensive understanding of the most common types of offences committed by juveniles and the trends in local areas. By analysing this data, policymakers can develop targeted strategies and policies to address the issue effectively.
# 
# - Ethics: Using the actual numbers of the news published to support that the concern is not personal feeling.
# - Reason for choosing the topic: Although the growing concern about youth crime in society, there are no indications that it is decreasing. It can still be seen that robbery or other offences of youth crime occurred frequently reveal by the news. That can be considered an urgent issue that needs to be solved.

# **Figure 1.**

# ![alt text](newsplot.png)

# ## General Question:
# How youth crime has changed over time in different local areas in Queensland? Additionally, are there certain types of offences that occur more frequently among youth compared to adults, and do these patterns differ in different local areas?
# 
# - Purpose: By answering this question, a comprehensive understanding of the temporal trends and changes in youth crime, and the specific offence types prevalent among youth crime can be concluded. This information can assist policymakers and communities in developing targeted interventions and strategies to address youth crime in different local areas.
# 
# - Ethics: Before answering this question, it is important to analyse all areas thoroughly, rather than just focusing on certain areas that may be got more concern. There are no assumptions here.
# 
# - Intended Audience: This can be everyone, not only the Queenslanders. This issue is not limited to a specific country, and therefore, the analysis findings can be shared with the world as valuable reference information.
# 

# ## Major Data Source
# 1. Queensland Police provide various open data on their [Maps and Statistics webpage](https://www.police.qld.gov.au/maps-and-statistics). For this research, the offences data including location information and offences type are needed. Furthermore, the information about whether the offenders are juveniles is also important. Therefore, `LGA Reported offenders number`, the dataset includes the major information we need, be selected as the data source .
# 
# 2. To understand the variation between local areas, visualisation can be important. Australian Government provides open data `QLD Local Government Areas - PSMA Administrative Boundaries GeoJSON(GEOJSON)` show the geographic information of each local govcernment in `GeoJSON file` on [data.gov.au](https://data.gov.au/dataset/ds-dga-16803f0b-6934-41ae-bf82-d16265784c7f/details?q=Queensland%20LGA%20boundaries).
# 
# - Ethics: The data contain all crime data in Queensland, including all sex, age, and all types of offences.
# - Source: All the data are from the government and all open data. 
# - Limitations: The crime data show the crime detail to local areas, not suburbs or even streets. The crime data did not show the date of a specific time. It is limited to local areas and monthly data from January 2001 to April 2023. 
# - Quality: All data quality is well with good columns name and no null data.
# - Amount: There are 125424 rows of data with 89 kinds of offences and 78 local areas.
# 

# ### Import packages

# In[14]:


# Import packages 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale


# ### LGA Reported offenders number

# #### Read data from URL
# The dataset `LGA Reported offenders number` is provided to the public on [Maps and Statistics webpage](https://www.police.qld.gov.au/maps-and-statistics). It can be downloaded with a specific URL.

# In[15]:


# Set the URL for the data source
url = 'https://open-crime-data.s3-ap-southeast-2.amazonaws.com/Crime%20Statistics/LGA_Reported_Offenders_Number.csv'


# `pd.read_csv` can be used to read a specific csv file. <br>
# `index_col=False`, use the headers as the columns names  
# 
# NOTE: The last column of this data is a number sequence and has no header, if `index_col=False` has been set will not read the last column and show a **ParserWarning**. Because the number sequence data was not essential, the **ParserWarning** can be ignored.

# In[16]:


# Read csv data from specific URL
df_lga = pd.read_csv(url, index_col=False)
df_lga.head(10)


# In[17]:


total_data = len(df_lga)
offence_type_count = len(df_lga.columns)-4
local_areas_count = len(df_lga['LGA Name'].unique())
print(f'There are {total_data} rows of data')
print(f'There are {offence_type_count} kinds of offences')
print(f'There are {local_areas_count} local areas')


# Most data can be identify the meaning by the column name, but `Other Offences`.
# According to [2016 - 2017 Annual Statistical Review](https://www.police.qld.gov.au/sites/default/files/2019-01/AnnualStatisticalReview_2016-17.pdf), `Other offences` include various offence types. Generally these offences detected by police rather than being reported by the public, such as Drug offences, traffic
# offences and good order offences.

# #### Data cleaning
# To make sure the data quality, data cleaning is an essential step.
# Firstly, remove or fill in the missing data. Secondly, convert the data with the wrong data type to the data type they should be.
# 

# In[18]:


# Check missing data
print('na data: ', df_lga.isna().sum().sum())
print('null data:', df_lga.isnull().sum().sum())


# There are no `na` or `null` data. But it may exist `whitespace` or `empty` data that can impact the data quality. `pandas.Series.value_counts` can show the unique values, that can be used to check `whitespace` or `empty`. Only `LGA Name`, `Month Year`, `Age`, `Sex` need be checked because only the string(object) type columes can include `whitespace` or `empty`.

# In[19]:


# Check district
print('whitespace or empty in LGA Name:',
      [v for v in df_lga['LGA Name'].value_counts().index if len(v.strip()) == 0])


# In[20]:


# Check Month Year
print('whitespace or empty in Month Year:',
      [v for v in df_lga['Month Year'].value_counts().index if len(v.strip()) == 0])


# In[21]:


# Check Month Year
print('whitespace or empty in Age:',
      [v for v in df_lga['Age'].value_counts().index if len(v.strip()) == 0])


# In[22]:


# Check Month Year
print('whitespace or empty in Sex:',
      [v for v in df_lga['Sex'].value_counts().index if len(v.strip()) == 0])


# --There are no missing data--

# In[23]:


# Check data type
# It can be simply split the columns into two group

# The columns that should be string (object):('LGA Name', 'Month Year', 'Age', 'Sex')
cols_str = list(df_lga.columns[:4])

# The columns that should be int: the others
cols_int = list(df_lga.columns[4:])


# In[24]:


# Show the object column in the wrong type
for column in df_lga[cols_str]:
    if df_lga[column].dtype != 'object':
        print(f"Column '{column}' is not type object.")


# In[25]:


# Show the int column in the wrong type
for column in df_lga[cols_int]:
    if df_lga[column].dtype != 'int64':
        print(f"Column '{column.dtype}' is not type int64.")


# --There are no wrong type data--<br>
# However, `Month Year` can be converted into `datetime` data type. Because it can be good for sorting, filtering, and comparing. 

# In[26]:


# A function to convert the specific format ('JAN01' = January, 2001) string to datetime
def convert_to_datetime(string):
    month = string[:3]
    year = string[3:]
    dt = datetime.strptime(month, '%b').replace(year=2000+int(year))
    return dt


# In[27]:


# Convert the Month Year by function convert_to_datetime
df_lga['Month Year'] = df_lga['Month Year'].apply(convert_to_datetime)


# In[28]:


# Check result
print(df_lga['Month Year'].dtype)
df_lga.head(10)


# In[29]:


min_date = df_lga['Month Year'].min()
max_date = df_lga['Month Year'].max()
print(f'The data is between {min_date.month}/{min_date.year} and {max_date.month}/{max_date.year}')


# --Data cleaning finish--

# ### QLD Local Government Areas - PSMA Administrative Boundaries GeoJSON(GEOJSON)

# #### Read data from URL
# 
# Download GeoJSON file `QLD Local Government Areas - PSMA Administrative Boundaries GeoJSON(GEOJSON)` from [data.gov.au](https://data.gov.au/dataset/ds-dga-16803f0b-6934-41ae-bf82-d16265784c7f/details?q=Queensland%20LGA%20boundaries).

# In[30]:


# url of GeoJSON of LGA area
url_geojson = 'https://data.gov.au/geoserver/qld-local-government-areas-psma-administrative-boundaries/wfs?request=GetFeature&typeName=ckan_16803f0b_6934_41ae_bf82_d16265784c7f&outputFormat=json'


# In[31]:


# Read GeoJSON file from url
response = requests.get(url_geojson)
geojson_data = response.json()


# In[32]:


# Read GeoJSON file into a pandas DataFrame
df_geo = pd.DataFrame(geojson_data, columns=['type', 'features'])


# In[33]:


# Extract local government name and the specific id from json file
df_geo['loca'] = df_geo['features'].apply(lambda x: x['properties']).apply(lambda x: x['qld_lga__2'])
df_geo['id_'] = df_geo['features'].apply(lambda x: x['id'])


# In[34]:


df_geo.head(10)


# #### Data cleaning
# Make sure the local government name of `df_geo` match to the major data `df_lga`

# In[35]:


# Convert the local government name in both dataframe to lower case

# Add 'council' to match df_lga
df_geo['loca'] = df_geo['loca'].str.lower() + ' council'
df_lga['LGA Name'] = df_lga['LGA Name'].str.lower()


# In[36]:


# Extract no repeated local government name from df_geo
geo_lga = df_geo['loca'].drop_duplicates().sort_values().reset_index(drop=True)


# In[37]:


mj_lga = df_lga['LGA Name'].drop_duplicates().sort_values().reset_index(drop=True)


# In[38]:


# Check the lenth are the same
check = 'yes' if len(mj_lga) == len(geo_lga) else 'No'
print(f'Is the two data have the same number of local government name: {check}')


# In[39]:


# Check any mismatch
count = 0
for i, name in enumerate(mj_lga):
    if name != geo_lga[i]:
        count+=1
        print(f'df_lga shows {name}, but df_geo shows {geo_lga[i]}')
print(f'There are {count} mismatch')


# There are several names are different between two data, we need map them manually

# In[40]:


# Create a mapping dictionart for local government name
name_mapping_dic = {}

name_mapping_dic['lockhart river aboriginal shire council']=  'lockhart river shire council'      
name_mapping_dic['napranum aboriginal shire council'      ]=  'napranum shire council'            
name_mapping_dic['palm island aboriginal shire council'   ]=  'palm island shire council'         
name_mapping_dic['woorabinda aboriginal shire council'    ]=  'woorabinda shire council'          
name_mapping_dic['mapoon aboriginal shire council'        ]=  'mapoon shire council'              
name_mapping_dic['yarrabah aboriginal shire council'      ]=  'yarrabah shire council'            
name_mapping_dic['doomadgee aboriginal shire council'     ]=  'doomadgee shire council'           
name_mapping_dic['wujal wujal aboriginal shire council'   ]=  'wujal wujal shire council'         
name_mapping_dic['cherbourg aboriginal shire council'     ]=  'cherbourg shire council'         
name_mapping_dic['pormpuraaw aboriginal shire council'    ]=  'pormpuraaw shire council'          
name_mapping_dic['blackall tambo regional council'        ]=  'blackall-tambo regional council'   
name_mapping_dic['hope vale aboriginal shire council'     ]=  'hope vale shire council'           
name_mapping_dic['kowanyama aboriginal shire council'     ]=  'kowanyama shire council'           


# In[41]:


# Map the LGA name of df_geo to df_lga
df_geo['LGA Name'] = df_geo['loca'].apply(lambda x: name_mapping_dic.get(x, x))


# In[42]:


# check again any mismatch
geo_lga = df_geo['LGA Name'].drop_duplicates().sort_values().reset_index(drop=True)
mj_lga = df_lga['LGA Name'].drop_duplicates().sort_values().reset_index(drop=True)
count = 0
for i, name in enumerate(mj_lga):
    if name != geo_lga[i]:
        count+=1
        print(f'df_lga shows {name}, but df_geo shows {geo_lga[i]}')
print(f'There are {count} mismatch')


# All data are ready and match to each other.<br>
# --Data clean done!--

# ## Question 1
# What are the temporal trends in youth crime across different local areas over time?
# - This question aims to analyse the variations in youth crime and identify the temporal trends. By understanding temporal trends across different local areas, we can gain insights into the local areas that can be helpful for policymakers in considering the resource allocation for each area and the prioritisation.

# ###### Auxiliary Functons for Data Analysis

# Build the functions to avoid doing repeat code

# In[43]:


def add_adult_juvenile(df, df_source, merge_cols):
    '''
    This function aims to create two new columns 'Adult' and 'Juvenile' on 'df',
    and merge the number from 'df_souce' into 'df' based on the 'merge_cols'
    '''
    lst_age = ['Adult', 'Juvenile']
    
    for age in lst_age:
        df = df.merge(df_source[df_source['Age'] == age][merge_cols + ['all']],
                                            on=merge_cols, how='left', suffixes=('', '_x'))
        df[age] = df['all_x']
        df.drop('all_x', axis=1, inplace=True)

    return df


# In[44]:


def add_analysis_cols(df):
    '''
    This function can create several extended columns from columns
    'all', 'Adult', and 'Juvenile' on the input dataframe, 
    these columns can show the difference between rows and the changing rate
    '''
    df['all_increase'] = df['all'].diff()
    df['all_increase_%'] = (df['all'].diff() / df['all'].shift(1)) * 100

    df['Juvenile_%'] = (df['Juvenile']/df['all'])*100
    df['Juvenile_increase'] = df['Juvenile'].diff()
    df['Juvenile_increase_%'] = (df['Juvenile'].diff() / df['Juvenile'].shift(1)) * 100

    df['Adult_%'] = (df['Adult']/df['all'])*100
    df['Adult_increase'] = df['Adult'].diff()
    df['Adult_increase_%'] = (df['Adult'].diff() / df['Adult'].shift(1)) * 100

    df.fillna(0, inplace=True)
    
    return df


# ### Question 1.1
# What are the temporal trends in youth crime compared to the other crime for the past decades?
# 

# ##### Data 

# There is one data might be used in answering this question:
# 1. The major data `df_lga` for analysing crime data

# In[45]:


# Major data
df_lga_q1 = df_lga.copy()


# ##### Analysis

# Firstly, to answer this question, we will focus on the difference between Adult and Juvenile. Therefore, we can scrap the `Sex` column by group up it. Then we can have an overview of temporal trends by years of the number of crime cases including adults and juveniles. 
# 
# - Ethics: To consider the overall situation, not take only some specific gender of the offender

# In[46]:


# Group the dataframe by columns except 'Sex'
df_lga_q1 = df_lga_q1.groupby(['LGA Name', 'Month Year', 'Age']).sum().reset_index()


# To have an overview of temporal trends by year, it was needed to sum up the number of different crimes by year. 

# In[47]:


# Create a column to show the number of all the crimes
df_lga_q1['all'] = df_lga_q1.sum(axis=1, numeric_only=True)


# In[48]:


# Extract year from datetime column 'Month Year'
df_lga_q1['year'] = df_lga_q1['Month Year'].dt.year


# In[49]:


# Sum up the data by year and age(adult, youth)
df_lga_q1_year = df_lga_q1.groupby(['year', 'Age']).sum().reset_index()


# For analysing the overall temporal trends, the data need to group by and order by year. Furthermore, to understand to changing levels by years, we can add columns about increasing numbers and rates on the data each year
# - Ethics: To separate the offender group into youth and adult, and keep the total data to compare each to avoid only analyse the youth crime data to lose the overall crime trend.

# In[50]:


# New a dataframe group by year
df_lga_q1_year_all = df_lga_q1_year.groupby('year').sum().reset_index()


# In[51]:


# Add adult and juvenile data to the data by year
df_lga_q1_year_all = add_adult_juvenile(df_lga_q1_year_all, df_lga_q1_year, ['year'])

# Add the columns about increasing numbers and rates on the data each year
df_lga_q1_year_all = add_analysis_cols(df_lga_q1_year_all)


# In[52]:


# List the columns we focus on
analysis_cols = ['Juvenile', 'Juvenile_%', 'Juvenile_increase', 'Juvenile_increase_%', 
                 'Adult', 'Adult_%', 'Adult_increase', 'Adult_increase_%',
                 'all', 'all_increase', 'all_increase_%']


# In[53]:


# Show the data
df_lga_q1_year_all = df_lga_q1_year_all[['year'] + analysis_cols]
df_lga_q1_year_all


# From this table, we can find the crime number was not increased every year. However, we can show the statistics data by using `describe()` to show the overall picture.
# 
# We need to remove the data in **2001** due to uncalculatable increasing data, and **2023** as well because that was not the data with a full year.

# In[54]:


# Filtering the data between 2002 and 2022, and showing the statistical data by using describe()
df_lga_q1_year_all[(df_lga_q1_year_all['year']>2001) & (df_lga_q1_year_all['year']<2023)].describe()


# Overall, from the mean of the increased number of youth, adult crimes, and the total number of offences, there is no significant increase but did growth due to it was positive. To find more information and the trend, we can focus on the recent 5 years (2023 not included).
# - Ethics: Analysing both the raw data and statistics data to avoid only analysing mean numbers to get wrong information.

# In[55]:


# Filtering the data in the past five years (2018 - 2023)
df_lga_q1_year_all_2018 = df_lga_q1_year_all[(df_lga_q1_year_all['year'] >= 2018) & (df_lga_q1_year_all['year'] < 2023)][['year'] + analysis_cols]
df_lga_q1_year_all_2018


# In[56]:


# Show the statistics data by using describe()
df_lga_q1_year_all_2018.describe()


# From the two tables above, it can be noticed youth crime cases increased significantly **11%** in **2022**, which is much more than the growth in adult crime in **2022 (3.7%)**. 
# 
# However, in the past 5 years, the most significant growth in overall crime and youth crime both happened **2022**, which were **41280(+5.1%)** and **16614(+11.6%)**, respectively. 
# 
# Furthermore, in the past 5 years, the average of all offence increasing number is **-6165**, but the number of youth crimes was **+3220**, it can be considered youth crime was growing in Queensland but not all crimes. 
# 
# - Ethic: Analysing both average data and the data by each year to avoid misunderstanding by only looking into the average number. 

# ##### Visualisation

# To show the temporal trend, a suitable diagram can be a line chart. Firstly, look into the data by year to perform an overview of the offences data including the total crime increasing percentage, adult crime increasing percentage, and youth crime increasing percentage in each **between 2002 and 2022**.
# 
# - Ethics: Compare the increasing rates instead of the case number and show each group and total increasing rate on the chart to indicate the differences. Because each group has a different population can cause the scale of the number of increasing differently which was unfair for comparing.
# 

# In[57]:


# Filter the data we need
df_linechart = df_lga_q1_year_all[(df_lga_q1_year_all['year']>2001) & (df_lga_q1_year_all['year']<2023)]


# In[58]:


# Build the line chart
fig_linechart = go.Figure()

# Add lines for Juvenile, Adult, and all
fig_linechart.add_trace(go.Scatter(x=df_linechart['year'], y=df_linechart['Juvenile_increase_%'], mode='lines', name='Juvenile'))
fig_linechart.add_trace(go.Scatter(x=df_linechart['year'], y=df_linechart['Adult_increase_%'], mode='lines', name='Adult'))
fig_linechart.add_trace(go.Scatter(x=df_linechart['year'], y=df_linechart['all_increase_%'], mode='lines', name='All'))

# Set the layout of the figure
fig_linechart.update_layout(title='Offences Trend by Year',
                            xaxis=dict(title='Year', type='category'),
                            yaxis=dict(title='Offences (%)'),
                            legend_title='Offender Groups')

# Show the plot
fig_linechart.show()


# From the above line chart, youth crime in Queensland has been growing continuously for three consecutive years since **2020** and the increasing rate is higher than adult crime, and there has been a significant increase in juvenile crime in **2022**.

# ##### insight

# After analysing temporal trends and patterns in youth crime across various local areas and compare to adult crime, we can conclude several key insights. Firstly, although youth crime has slightly increased over time, it has not risen significantly more than adult crimes in the past 20 years. However, focusing on recent years, specifically in **2022**, shows a noticeable increase in **youth crime (11% rise)** compared to **adult crime (3.7% rise)**. This indicates that youth crime is becoming a more pressing issue in Queensland.

# ### Question 1.2
# What are the temporal trends in youth crime compared to the other crime since 2022?
# - After analysing the overall trend in the past decades, we can look into each month from **2022** to compare the youth crime cases between **2022** and **2023** to identify are the youth crime number still increasing.
# 

# ##### Data 

# There is one data might be used in answering this question:
# 1. The major data `df_lga` for analysing crime data, in here we can continuously keep using the `df_lga_q1`.

# ##### Analysis

# We will focus on the recent monthly data between youth crime and adult crime. Therefore, we can group the data by month and filter the data between **2022** and **2023**. Then we can have an overview of temporal trends by month of the number of crime cases including adults and juveniles. 

# In[59]:


# observe the data
df_lga_q1.head(10)


# In[60]:


# Group up the data by month
df_lga_q1_month = df_lga_q1.groupby(['Month Year', 'Age']).sum().reset_index()
df_lga_q1_month_all = df_lga_q1.groupby(['Month Year']).sum().reset_index()


# In[61]:


# Add adult and juvenile data to the data by month
df_lga_q1_month_all = add_adult_juvenile(df_lga_q1_month_all, df_lga_q1_month, ['Month Year'])

# Add the columns about increasing numbers and rates on the data each month
df_lga_q1_month_all = add_analysis_cols(df_lga_q1_month_all)


# In[62]:


# Filtering the data between Jan 2022 to Mar 2023, not consider Apr 2023 due to data may incomplete
df_lga_q1_month_all_2022 = df_lga_q1_month_all[(df_lga_q1_month_all['Month Year'] >= '2022-01-01')
                                               & (df_lga_q1_month_all['Month Year'] < '2023-04-01')][['Month Year'] + analysis_cols]

df_lga_q1_month_all_2022


# In[63]:


# Show the statistics data by using describe()
df_lga_q1_month_all_2022.describe()


# From **2022**, despite **Mar 2023** was not with the highest number of youth crime cases, it has the highest increasing percentage **25%** and increasing number **2930**. Compare to the audlt crime in **Mar 2023**, the growing percentage was only **5%** which was much less than in youth crime.
# 
# - Ethics: Compare each specific month and group to avoid using average numbers. Because the average number can be affected by some significant numbers can cause misunderstanding.

# ##### Visualisation

# To show the temporal trend, a suitable diagram can be a line chart. We can show the data by month to perform an overview of the offences data including the total crime increasing percentage, adult crime increasing percentage, and youth crime increasing percentage in each **between Jan 2022 and Mar 2023**.
# 
# - Ethics: Compare the increasing rates instead of the case number and show each group and total increasing rate on the chart to indicate the differences. Because each group has a different population can cause the scale of the number of increasing differently which was unfair for comparing.
# 

# In[64]:


# Filter the data we need
df_linechart_2 = df_lga_q1_month_all[(df_lga_q1_month_all['Month Year'] >= '2022-01-01')
                                     & (df_lga_q1_month_all['Month Year'] < '2023-04-01')][['Month Year'] + analysis_cols]


# In[65]:


# Convert the datetime format to mm-yyyy
df_linechart_2['month'] = df_linechart_2['Month Year'].dt.strftime('%m-%Y')


# In[66]:


# Build the line chart
fig_linechart_2 = go.Figure()

# Add lines for Juvenile, Adult, and all
fig_linechart_2.add_trace(go.Scatter(x=df_linechart_2['month'], y=df_linechart_2['Juvenile_increase_%'], mode='lines', name='Juvenile'))
fig_linechart_2.add_trace(go.Scatter(x=df_linechart_2['month'], y=df_linechart_2['Adult_increase_%'], mode='lines', name='Adult'))
fig_linechart_2.add_trace(go.Scatter(x=df_linechart_2['month'], y=df_linechart_2['all_increase_%'], mode='lines', name='All'))

# Set the layout of the figure
fig_linechart_2.update_layout(title='Offences Trend by Month',
                              xaxis=dict(title='Month', type='category'),
                              yaxis=dict(title='Offences (%)'),
                              legend_title='Offender Groups')

# Show the plot
fig_linechart_2.show()
                    


# After looking into monthly data in the recent month (2022 and Jan, Feb, and Mar 2023), there are two specific months that had significant growth in youth crime which are **Nov 2022** and **Mar 2023**. Both months had a higher growth rate in youth crime than adult crime, which means the trend of youth crime did not exactly follow the overall crime. If we look into the numbers in `df_lga_q1_month_all_2022`, it can be found that the highest youth crime number was in **Nov 2022** which was **16409** cases. However, **Mar 2023** was the month with the highest increasing rate of youth crime despite it had not the highest number of youth crime cases.
# 

# ##### Insight

# After analysing monthly data from 2022, there was no significant sign that shows which group grow much more, but the data and the line chart did not show any sign that the youth crime or overall offences are mitigating, and it seems the youth group with a bigger variation of the monthly data than in the adult group. Moreover, youth crime increased most in the most recent month **Mar 2023** in the data we collected, and the growth rate was **25%** which was much more the in adult crime which was **5%**.
# 

# ### Question 1.3
# What are the recent trends in youth crime across different local areas in Queensland?
# - For getting more information on the relationship between youth crime and local areas recently, we can compare the cases of youth crime in each local area. To understand to overall picture recently by the local area, we can pick **Jan 2023**, **Feb 2023**, and **Mar 2023** to analyse.

# ##### Data 

# There are two data might be used in answering this question:
# 1. The major data `df_lga` for analysing crime data, in here we can continuously keep using the `df_lga_q1`.
# 2. The GeoJSON data `df_geo` fro visualisation 

# ##### Analysis

# To focus on the recent three months **Jan 2023**, **Feb 2023**, and **Mar 2023**, we need to pick the data of these three months from the major data. Then group the data by month and local areas to observe the data and compare each group in local areas.

# In[67]:


# Need to include Dec 2022 to calculate the increasing number of Jan 2023 
df_lga_q1_2023 = df_lga_q1[(df_lga_q1['Month Year']>='2022-12-01') & (df_lga_q1['Month Year']<'2023-04-01')][['LGA Name', 'Month Year', 'Age', 'all']]


# In[68]:


#Group by local area and month
df_lga_q1_lga = df_lga_q1_2023.groupby(['LGA Name', 'Month Year']).sum().reset_index()


# In[69]:


# Add adult and juvenile data to the data by local area and month
df_lga_q1_lga = add_adult_juvenile(df_lga_q1_lga, df_lga_q1_2023, ['LGA Name', 'Month Year'])

# Add the columns about increasing numbers and rates on the data
df_lga_q1_lga = add_analysis_cols(df_lga_q1_lga)


# In[70]:


# Remove Dec 2022 to focus on 2023
df_lga_q1_lga = df_lga_q1_lga[df_lga_q1_lga['Month Year'] >= '2023-01-01'].reset_index(drop=True)


# In[71]:


# Show the data
df_lga_q1_lga.head(10)


# To understand the rank of each local area, we can calculate the average number of the three months by each local area and sort it, then pick the top 5 local areas by different numbers to show the detail and compare them to identify which area has a more serious crime issue
# 
# - Ethics: Separate the data by different areas and groups to avoid information loss after calculating the average, and We can understand the overall seasonal trend by mean.

# Firstly, we focus on the overall crime

# In[72]:


# Group by local area
df_lga_q1_lga_group = df_lga_q1_lga.groupby('LGA Name').mean()


# In[73]:


# Sort by overall crime number
df_lga_q1_lga_group.sort_values('all', ascending=False).head(5)[['all', 'all_increase', 'all_increase_%']]


# In[74]:


# Sort by increasing number
df_lga_q1_lga_group.sort_values('all_increase', ascending=False).head(5)[['all', 'all_increase', 'all_increase_%']]


# From the two tables above, despite **Brisbane** and **Gold Coast** having the highest number of offences in the first three months of 2023, both numbers in these two areas were decreasing which are **-243** and **-112**, respectively. If we look into the average increasing number, it can be noticed that **Logan** has the largest increasing number of overall crimes. Also, it can be noticed that **Rockhampton** and **Mackay** did not on the list of top 5 crime numbers but were on the top 5 increasing crime list. It can be considered the offences cases grow significantly in these two areas.

# Secondly, we focus on the youth crime

# In[75]:


# Sort by Juvenile crime number
df_lga_q1_lga_group.sort_values('Juvenile', ascending=False).head(5)[['Juvenile', 'Juvenile_increase', 'Juvenile_increase_%']]


# In[76]:


# Sort by Juvenile crime increasing number
df_lga_q1_lga_group.sort_values('Juvenile_increase', ascending=False).head(5)[['Juvenile', 'Juvenile_increase', 'Juvenile_increase_%']]


# Looking into youth crime, **Brisbane** also has the highest number of youth crime cases, but the number was decreasing in the past few months. 
# 
# For the increasing number, **Cairns** seems to have a significant growth in youth crime because it had the highest number of increasing numbers, and it was the second ranking area of the top 5 youth crime number list.
# 
# Furthermore, the number of youth crime cases in **Mornington** and **Palm Island** seems to be almost doubled since 2023, the mean of the percentage of increasing numbers were **82%** and **104%**, respectively.
# 
# - Ethics: Compare the ranks by sorting the case number and the increasing number, can avoid some areas with a significantly high increasing percentage due to a population too small to affect the ranking.

# ##### Visualisation

# To understand the crime number in each local area, the better way to visualise the data is to show the data on a map and present the number in different colors to highlight the difference. Firstly, we can show the overall situation of all the offences in each local area on the map by using `choropleth_mapbox`. To focus on the situation recently, we use the data `df_lga_q1_lga_group` which includes the data from **Jan 2023 to Mar 2023**.
# 
# - Ethics: To show four different ranks by local areas to have a comprehensive understanding of the trend of each local area and the geographic distribution to make sure the data in each area can be seen.

# In[77]:


# Mergo the GeoJSON data into dataframe with crime data
df_map_q1 = df_geo.merge(df_lga_q1_lga_group, left_on='LGA Name', right_on='LGA Name', how='left')


# In[78]:


# Set the center of the map
latitude = -20
longitude = 145


# In[79]:


# Create a choropleth map
fig_map_all = px.choropleth_mapbox(df_map_q1, 
                                   geojson=geojson_data, 
                                   locations='id_',
                                   hover_name = "LGA Name",
                                   hover_data=['all', 'all_increase', 'Adult', 'Juvenile'],
                                   color = 'all',
                                   mapbox_style="open-street-map",
                                   color_continuous_scale='ylorrd',
                                   opacity = 0.5)

fig_map_all.update_layout(
    title ='Map 1. Number of Offences in Local Areas',
    mapbox_center_lat = latitude, 
    mapbox_center_lon = longitude, 
    mapbox_zoom = 3.3,
    margin={"r":0,"t":50,"l":0,"b":0},
    coloraxis_colorbar=dict(
        title='All Offences Number'  # Set the label for the color legend
    )
)

# Display the map
fig_map_all.show()


# From the above map, it can be observed that there are only a few specific local areas, such as **Cairns**, **Townsville**, and **the area around Brisbane** including **Brisbane**, which had significantly higher crime numbers than other areas. However, generally, densely populated areas should have a higher number of crime cases. To find more information about the crime trend in each area, we can show the increased difference by each local area on the map.

# To focus on the increasing number of crimes, set the decreasing number to 0

# In[80]:


# Set the decreasing number to 0
df_map_q1['all_increase_'] = df_map_q1['all_increase'].apply(lambda x: x if x > 0 else 0)


# In[81]:


# Create a choropleth map
fig_map_all = px.choropleth_mapbox(df_map_q1, 
                                   geojson=geojson_data, 
                                   locations='id_',
                                   hover_name = "LGA Name",
                                   hover_data=['all', 'all_increase', 'Adult', 'Juvenile'],
                                   color = 'all_increase_',
                                   mapbox_style="open-street-map",
                                   color_continuous_scale='ylorrd',
                                   opacity = 0.5)

fig_map_all.update_layout(
    title ='Map 2. Increasing Number of Offences in Local Areas',
    mapbox_center_lat = latitude, 
    mapbox_center_lon = longitude, 
    mapbox_zoom = 3.3,
    margin={"r":0,"t":50,"l":0,"b":0},
    coloraxis_colorbar=dict(
        title='Increasing Number'  # Set the label for the color legend
    )
)

# Display the map
fig_map_all.show()


# The map above shows the different information from the last map, it presents the increasing level in each local area which can show the crime trend of each area. Firstly, **Brisbane** did not have the highest number, instead **Logan**. Secondly, there are several local areas that have notable growth in offences number but not significant crime numbers shown in `map 1` such as **Mackay**, **Rockhampton**, **Bundaberg**, and **Fraser Coast**.

# To understand more about youth crime, we need to show only the youth crime number of each local area on the map. 

# In[82]:


# Create a choropleth map
fig_map_youth = px.choropleth_mapbox(df_map_q1,
                                     geojson=geojson_data, 
                                     locations='id_',
                                     hover_name = "LGA Name", 
                                     hover_data=['all', 'Adult', 'Juvenile', 'Juvenile_increase'],
                                     color = 'Juvenile',
                                     mapbox_style="open-street-map",
                                     color_continuous_scale='ylorrd',
                                     opacity = 0.5)

fig_map_youth.update_layout(
    title ='Map 3. Number of Youth Crime in Local Areas',
    mapbox_center_lat = latitude, 
    mapbox_center_lon = longitude, 
    mapbox_zoom = 3.3,
    margin={"r":0,"t":50,"l":0,"b":0},
    coloraxis_colorbar=dict(
        title='Youth Crime Number'  # Set the label for the color legend
    )
)

# Display the map
fig_map_youth.show()


# This map shows similar information to `map 1`, the local area, such as **Brisbane**, **Cairns**, and **Townsville** had seriously higher youth crime numbers than other areas. However, we need to find more details about the youth crime trend in local areas. Therefore, we can show the increasing number of youth crimes instead of just the offences number.

# To focus on the increasing number of youth crimes, set the decreasing number to 0

# In[83]:


df_map_q1['Juvenile_increase_'] = df_map_q1['Juvenile_increase'].apply(lambda x: x if x > 0 else 0)


# In[84]:


# Create a choropleth map
fig_map_youth = px.choropleth_mapbox(df_map_q1,
                                     geojson=geojson_data, 
                                     locations='id_',
                                     hover_name = "LGA Name",
                                     hover_data=['all', 'Adult', 'Juvenile', 'Juvenile_increase'],
                                     color = 'Juvenile_increase_',
                                     mapbox_style="open-street-map",
                                     color_continuous_scale='ylorrd',
                                     opacity = 0.5)

fig_map_youth.update_layout(
    title ='Map 4. Increasing Number of Youth Crime in Local Areas',
    mapbox_center_lat = latitude, 
    mapbox_center_lon = longitude, 
    mapbox_zoom = 3.3,
    margin={"r":0,"t":50,"l":0,"b":0},
    coloraxis_colorbar=dict(
        title='Increasing Number'  # Set the label for the color legend
    )
)

# Display the map
fig_map_youth.show()


# From the map, we can find **Cairns** has the most serious youth crime increasing trend. Also the area around **Cairns** such as **Douglas**, **Mareeba**, and **Tablelands** have a notable increasing trend to some degree. For **Mackay**, it has been noticed that both the increasing number in overall crime and youth crime have a notable number from other areas. Moreover, **Sunshine Coast** and **Toowoomba** can be noticed because it seems to have a medium degree of youth crime increasing trend but not an overall crime trend.

# ##### Insight

# From the analysis of youth crime across various local areas, there are some areas that are noticeable. **Cairns** is one of the areas where youth crime has significantly increased, and this trend is also observed in nearby areas such as **Douglas**, **Mareeba**, and **Tablelands** but the increased numbers were not as high as in **Cairns**. Additionally, other local areas like **Mackay**, **Sunshine Coast**, and **Toowoomba** have also experienced a medium degree of youth crime increasing trend.

# ##### Memory release

# In[85]:


# delete the dataframe onyl for q1 to release memory
del df_lga_q1
del df_lga_q1_2023
del df_lga_q1_lga
del df_lga_q1_lga_group
del df_lga_q1_month
del df_lga_q1_month_all
del df_lga_q1_month_all_2022
del df_lga_q1_year
del df_lga_q1_year_all
del df_lga_q1_year_all_2018
del df_map_q1


# ## Question 2
# Are there any specific types of offences that are more common among youth offenders compared to adult offenders, and any differences between local areas?
# 
# - This question focuses on analysing the differences between the common types of offences in youth crime and adult crime. Understanding the specific type of offences among youth crime in different local areas can help in planning intervention programs which can against and addressing the specific types of crimes by areas.

# ### Question 2.1
# What are the differences between youth crime and adult crime for the common offence types?
# 
# - To understand if is there any specific offence type common among youth offenders but not in adult crime.

# ##### Data 

# There is one data might be used in answering this question:
# 1. The major data `df_lga` for analysing crime data

# In[86]:


# Major data
df_lga_q2 = df_lga.copy()


# ##### Analysis

# To understand the common offences type among youth offenders and adult offenders, we can start from split the data into two groups. Also, we can focus on the recent data due to this is an current issue. 
# 
# Firstly, we can filter the recent data, which can be the data after 2022 because according the result from question 1, 2022 had the largest growth in youth crime and overall crime in the past 5 years. 
# 
# - Ethics: Based on the previous result choose the most valuable data for analysis

# In[87]:


# Filter the data
df_lga_q2 = df_lga_q2[df_lga_q2['Month Year'] >= '2022-01-01']


# Then we can group up the data by offender(`Age`) to analyse the offences types

# In[88]:


# Group by Age and sum up the number
df_lga_q2_group = df_lga_q2.groupby('Age').sum()
df_lga_q2_group


# To analyse the offences types, we need to sort the number of each offences type.
# One of the methods to sort it is that we can pivot the dataframe by calling `transpose()`. Then we can compare the distribution of the different offences types in the two groups. 

# In[89]:


# Pivot the dataframe
df_lga_q2_group_pivot = df_lga_q2_group.transpose()
df_lga_q2_group_pivot


# In[90]:


# Add ranking columns for two groups to show the ranks of each offences type
df_lga_q2_group_pivot['ranking_adult'] = df_lga_q2_group_pivot['Adult'].rank(ascending=False).astype('int')
df_lga_q2_group_pivot['ranking_juvenile'] = df_lga_q2_group_pivot['Juvenile'].rank(ascending=False).astype('int')


# In[91]:


# Sort by youth crime
df_lga_q2_group_pivot.sort_values('ranking_juvenile').head(10)[['ranking_adult', 'ranking_juvenile']]


# In[92]:


# Sort by adult crime
df_lga_q2_group_pivot.sort_values('ranking_adult').head(10)[['ranking_adult', 'ranking_juvenile']]


# From the above two tables, it can be found there is a significant difference in prevalent offences types between the two groups. In the top 10 common crime offences type in the two groups, there are only 5 offences that are the same which are **Other offences**, **Offences Against Property**, **Other Theft (excl. Unlawful Entry)**, **Offences Against the Person**, and **Drug Offences**. 
# 
# The offences related to unlawful entry include **Unlawful Entry**, **Unlawful Entry With Intent - Dwelling**, and **Unlawful Entry Without Violence - Dwelling** are all prevalent offences in youth crime but not in adult crime. Furthermore, **Unlawful Use of Motor Vehicle** is also one of the crime types that common in youth crime but adult. 
# 
# - Ethics: Compare the common offence types of each group by sorting each group. It can show the complete top 10 crime types of each group.

# ##### Visualisation

# To visualise the comparing of the prevalent offences among adult and juvenile crime, we can use a bar chart to show two groups in two bars. Because the number of total offence types are 89, which may not be a suitable number to show all the offence types. Therefore, we pick the top **20** common crime types among youth crime to show in bar chart to compare with adult crime.
# 
# - Ethics: Show the different groups in one chart and avoid using the same scale for confusing understanding. Set two y-axis on two sides can show the real situation in each group

# In[93]:


# Pick the top 20 common offence types in youth crime 
df_lga_q2_group_vis = df_lga_q2_group_pivot.sort_values('ranking_juvenile').head(20)


# In[94]:


# Create two bar graph objects 
bar1 = go.Bar(name='Juvenile Offence number', x=df_lga_q2_group_vis.index, y=df_lga_q2_group_vis['Juvenile'], yaxis='y', offsetgroup=1)
bar2 = go.Bar(name='Adult Offence number', x=df_lga_q2_group_vis.index, y=df_lga_q2_group_vis['Adult'], yaxis='y2', offsetgroup=2)

# Combine two bars
fig = go.Figure(
    data=[bar1, bar2],
    layout={'yaxis': {'title': 'Juvenile Offence number'},
            'yaxis2': {'title': 'Adult Offence number','overlaying': 'y', 'side': 'right'}
    }
)

# Layout setting
fig.update_layout(
    title_text="Offence Number in Adult and Youth Crime by Offence Types",
    width=16*60, height=9*60,
    legend_title='Offender Groups',
    legend=dict(x=1.2),
    barmode='group'
)

fig.show()


# From the above bar chart, we can find the differences between youth crime and adult crime. Firstly, crimes related to unlawful entry such as **Unlawful Entry**, **Unlawful Use of Motor Vehicle**, **Unlawful Entry With Intent - Dwelling**, and **Unlawful Entry Without Violence - Dwelling** are much more common in youth crime. However, in youth crime, drug-related offences like **Drug Offences** and **Other Drug Offences** are not as prevalent in adult crime.

# ##### Insight

# Among the top 10 common offenses between youth crime and adult crime, only five offenses are shared by both groups, which are **Other offences**, **Offences Against Property**, **Other Theft (excl. Unlawful Entry)**, **Offences Against the Person**, and **Drug Offences**.
# 
# In youth crime, Offenses related to unlawful entry, including **Unlawful Entry**, **Unlawful Entry With Intent - Dwelling**, and **Unlawful Entry Without Violence - Dwelling**, are more prevalent compared to adult crime. Additionally, the **offense of Unlawful Use of Motor Vehicle** is also more common in youth crime. On the other hand, drug-related offenses, such as **Drug Offenses** and **Other Drug Offenses**, are not as prevalent in youth crime as they are in adult crime.

# ### Question 2.2
# What are the differences between local areas for the common offence types in youth crime?
# - By analysing each local area, it can conclude the different common offence types by area which can help design interventions by area.

# ##### Data 

# There are two data might be used in answering this question:
# 1. The major data `df_lga` for analysing crime data, in here we can continuously keep using the `df_lga_q2`.
# 2. The GeoJSON data `df_geo` fro visualisation 

# ##### Analysis

# One of the methods to show the difference between these areas is that we can simply split them into several group by using `K-means`. Then analyse the features of each group. Firstly, we can use `Elbow Method` to find the suitable cluster number. Next can analyse these clusters. 
# 
# - Ethics: Use `Elbow Method` to find the suitable cluster number to avoid choosing a too large or too small number to split the data into no meaning groups.

# To determine the prevalent offences type in youth crime whether different between local areas, we can analyse the data by look into local areas. To focus on youth crime, we can filter the data with only youth crime data.

# In[95]:


# Group by 'LGA Name' and filter data to keep only 'Juvenile'
df_lga_q2_lga = df_lga_q2[df_lga_q2['Age'] == 'Juvenile'].groupby(['LGA Name']).sum()


# In[96]:


print(f'There are {len(df_lga_q2_lga)} local areas')


# Some area may have 0 offences, these area cannot be used for ranking need to be excluded

# In[97]:


# Sum the tital offences number for each area
df_lga_q2_lga['all'] = df_lga_q2_lga.sum(axis=1, numeric_only=True).astype('int')


# In[98]:


# Exclude the area has 0 offence number
df_lga_q2_lga = df_lga_q2_lga[df_lga_q2_lga['all']>0]


# In[99]:


print(f'New there are {len(df_lga_q2_lga)} local areas')


# In[100]:


# The column 'all' can be drop, it only for remove the 0 offence number area
df_lga_q2_lga = df_lga_q2_lga.drop('all', axis=1)


# Before starting clustering, normalisation is needed due to the different scales in each local area. This number after normalisation can be considered the probability of each offence type because it divided by the sum of the all offences in each city.
# 
# - Ethics: Do normalisation to keep every group is fair and can be compared on the same scale

# In[101]:


# Normalise the data by rows
normalised_df = df_lga_q2_lga.div(df_lga_q2_lga.sum(axis=1), axis=0)


# In[102]:


# Examine the result of normalisation by choosing one city 
normalised_df[normalised_df.index=='brisbane city council'].transpose().sort_values('brisbane city council', ascending=False).head(10)


# use `Elbow Method` to show the suitable number of cluster

# In[103]:


weights = []
lst_k = list(range(1, 10))

# Calculate the score of cluster from 1 to 9
for k in list(range(1, 10)):
    km = KMeans(n_clusters=k, random_state=0).fit(normalised_df)
    weights.append(km.inertia_) # Sum of squared distances of samples to their closest cluster center

# Show the weights of each k
fig_em = px.line(x=lst_k,
                 y=weights,               
                 markers = True,
                 width=9*60, 
                 height=6*60)

fig_em.update_layout(xaxis_title = 'Number of clusters',
                     yaxis_title = 'Sum of squared distances',
                     title = 'Elbow Method')
fig_em.show()


# From the above `Elbow Method` figur, there are no obvious 'elbow'. However, the poinit most likely an 'elbow' is 3, we can consider `cluster number = 3` can be a suitable number. Then we can start training the `K-means` model to produce the clusters.

# In[104]:


# Using K-Means fit the data
km3 = KMeans(n_clusters=3, random_state=0).fit(normalised_df)
print('The local areas in each group:')
print(pd.Series(km3.labels_).value_counts())


# In[105]:


# Put the cluster number back to the dataframe
normalised_df['cluster'] = km3.labels_.astype('str')


# After clustering, we can obverse the difference between each group. We can list the most common offence types in each group by calculating the average of the probability of each offence type in each group and sorting it to understand more detail about each group.

# In[106]:


# Take the areas name by clusters
lst_cluster_0 = list(normalised_df[normalised_df['cluster']=='0'].index)
lst_cluster_1 = list(normalised_df[normalised_df['cluster']=='1'].index)
lst_cluster_2 = list(normalised_df[normalised_df['cluster']=='2'].index)


# In[107]:


# Show the most common offence types in the group
cnm_type = []
for i, c in enumerate([lst_cluster_0, lst_cluster_1, lst_cluster_2]):
    
    sr = normalised_df[normalised_df.index.isin(c)][df_lga_q2_lga.columns].mean().sort_values(ascending=False).head(10)
    
    print(f'The most common offence types in the group {i}')
    print(sr)
    print('The local areas in this grouop:')
    print(c)
    print('----------------------------------------------------------\n')
    
    # Save the offence types by clusters for visualisation
    cnm_type.append("\n     "+"\n     ".join(list(sr.index)))


# 
# In **Group 0**, the most common offence types are **other offences**, **trespassing and vagrancy**, **public nuisance**, and **good order offences**. Seems there are no more other offences happening in these areas.
# 
# In **Group 1** and **Group 2**, both groups share some common offence types, but still different in the frequencies of specific offence types. **Group 1** is more focused on **property related offences** such as **Unlawful Entry With Intent - Other** and **Unlawful Entry With Intent - Dwelling**. **Group 2** had a broader range of offence types, including **property related crimes**, **other offences**, and **Drug offences**.
# 

# ##### Visualisation

# From the previous analysis, the local areas have been group into 3 clusters by using `K-means`. It can be show on the map for an overall picture of the distribution of the clusters.
# 
# - Ethics: To show all local areas on the map to have a comprehensive understanding of the geographic distribution and make sure the data in each area can be seen.

# In[108]:


# Create a dataframe to save the common offences of each cluster
d = {'cluster': ['0', '1', '2'], 'cnm_type': cnm_type}
df_cnm_type = pd.DataFrame(data=d)


# In[109]:


# Reset index for merging
df_normalised_map = normalised_df.reset_index()


# In[110]:


# Merge geojson data with crime data
df_normalised_map['LGA Name'] = df_normalised_map['LGA Name'].str.lower()
df_map_q2 = df_geo.merge(df_normalised_map, left_on='LGA Name', right_on='LGA Name', how='left')


# There are three area have no cluster due to no offence number, neen to exclude because no meaningful

# In[111]:


# Exclude the null cluster
df_map_q2 = df_map_q2[~df_map_q2['cluster'].isna()]


# In[112]:


# Create a choropleth map
fig_map_cluster = px.choropleth_mapbox(df_map_q2,
                                       geojson=geojson_data, 
                                       locations='id_',
                                       hover_name = "LGA Name",
                                       color = 'cluster',
                                       mapbox_style="open-street-map",
                                       opacity = 0.5,
                                       color_discrete_map={'0':'green',
                                                           '1':'red',
                                                           '2':'blue'}
                                      )
# Layout setting
fig_map_cluster.update_layout(
    title ='Map 5. The Local Area Clusters with Different Common Offence Types among Youth Crime',
    mapbox_center_lat = latitude, 
    mapbox_center_lon = longitude, 
    mapbox_zoom = 3.3,
    margin={"r":0,"t":50,"l":0,"b":0}
)

# Show the map
fig_map_cluster.show()
print('note: The white areas are the local areas with 0 youth crime offence')

# Show the common offence types of each cluster
print('The common offence types of each cluster:')

for index, row in df_cnm_type.iterrows():
    print('Cluster ', row['cluster'], ':')
    print(row['cnm_type'])
    print('----------------------------------------------------------\n')


# From this map, all local areas can be categorised into three groups which are **Group 0**, **Group 1**, and **Group 2**. We can find that most areas fall into **Groups 1 and 2**. Furthermore, from geographic location, it seems most local areas in **Group 2** are located on the east side and south side. On the other hand, the local areas in **Group 1** are located on the west side in general. 

# ##### Insight

# By analysing the offences types in local areas, we can categorise them into three groups: Group 0, Group 1, and Group 2. **Group 0** has noticeably fewer offenses compared to the other two groups. **Group 1** and **Group 2** have some common offence types, such as **Offenses Against Property**, **Other Offenses**, and **Other Property Damage**. However, **Group 1** is more focused on **property-related offenses**, like **Unlawful Entry With Intent - Other** and **Unlawful Entry With Intent - Dwelling**. On the other hand, **Group 2** has a wider range of offence types, including **property-related crimes**, **Assault** and **drug offenses**.
# 

# ## Summary

# Based on the analysis of youth crime over time in different local areas in Queensland, several key insights can be found. Overall, youth crime has not risen significantly or grown faster than adult crimes in the past two decades. However, the recent data in 2022 indicates a noticeable rise in youth crime (11% increase) compared to adult crime (3.7% increase) in the same year, it is being suggested that youth crime is becoming a growing concern in Queensland recently.
# 
# When analysing monthly data from 2022, there is no significant difference in growth between youth and adult crime. However, the youth crime cases decreased by 24% in Feb 2023, but it increased by 25% in the next month Mar 2023. Compared to adult crime, the growth rates are -3% and +5% in Feb 2023 and Mar 2023, respectively. It can be concluded that the youth crime data shows more variation in a monthly crime increasing rate compared to the adult group. 
# 
# Analysing youth crime across all local areas in Queensland reveals some specific areas had a significant increase. Cairns stands out as an area where youth crime has significantly increased, with similar trends observed in neighboring areas like Douglas, Mareeba, and Tablelands, although the growth rate was not as high as in Cairns. Additionally, some areas such as Mackay, Sunshine Coast, and Toowoomba have experienced a medium degree increasing in youth crime.
# 
# In terms of offence types, there are five common offences shared between youth and adult crime: Other offences, Offences Against Property, Other Theft (excluding Unlawful Entry), Offences Against the Person, and Drug Offences. However, offences related to unlawful entry, including Unlawful Entry, Unlawful Entry With Intent - Dwelling, and Unlawful Entry Without Violence - Dwelling are more prevalent in youth crime compared to adult crime. The offence of Unlawful Use of Motor Vehicle is also more common among youth. On the other hand, drug-related offences, such as Drug Offences and Other Drug Offences, are less common in youth crime compared to adult crime.
# 
# By analysing all the offence types in different local areas allows clustering into three groups: Group 0, Group 1, and Group 2. Group 0 has significantly fewer offences compared to the other two groups. Group 1 and Group 2 share some common offence types, including Offences Against Property, Other Offences, and Other Property Damage. However, Group 1 is more focused on property-related offences, such as Unlawful Entry With Intent - Other and Unlawful Entry With Intent - Dwelling. In contrast, Group 2 shows a broader range of offence types, including property-related crimes, Assault, and drug offences.
# 
# In conclusion, the analysis result indicates that youth crime in Queensland has shown some significant increase in specific months or particular areas but not in general over time. However, there is no sign showing that youth crime is mitigating. To plan more interventions on it can be necessary. Secondly, certain offence types, such as unlawful entry and motor vehicle-related offences, are more prevalent among youth. Local areas like Cairns, along with other regions such as Douglas, Mareeba, and Tablelands, have experienced notable increases in youth crime recently. These insights can inform policymakers and communities in developing targeted interventions and strategies to mitigate youth crime in different local areas. Furthermore, it can be based on the area groups to plan different policies and the local governments can share some experience with the other local councils who are in the same group. Based on the result of the analysis, these policies' direction can be helpful for address this growing concern issue. 
# 

# In[ ]:




