#!/usr/bin/env python
# coding: utf-8

# # 1.<span style="color:red"> Import des librairies </span>

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
import pycountry
import plotly.express as px
import altair as alt
import seaborn as sns
import statsmodels.stats.api as sms
import statsmodels.api as sm
from statsmodels.tsa.stattools import kpss
from scipy import stats
import scipy
import statsmodels
import sklearn
from sklearn.metrics import mean_squared_error
import statsmodels.stats.diagnostic as sms
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.model_selection import train_test_split
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.regression.linear_model import yule_walker
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, aic
import arch
from arch.unitroot import engle_granger
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import time
import warnings
import random
warnings.filterwarnings('ignore')

print ("%-20s %s"% ("numpy", np.__version__))
print ("%-20s %s"% ("pandas", pd.__version__))
print ("%-20s %s"% ("statsmodels", statsmodels.__version__))
print ("%-20s %s"% ("scipy", scipy.__version__))
print ("%-20s %s"% ("sklearn", sklearn.__version__))

pd.set_option('max_colwidth', 1000)


# # 2.<span style="color:red"> Lectures des données </span>

#  - ### 2.1 <span style="color:blue">  Metadata du continent Européen</span>

# In[2]:


df_meta = pd.read_csv("data/continents2.csv", header = 0, usecols=[0,5])
df_meta = df_meta.loc[df_meta["region"] == "Europe"]


#  - ### 2.2 <span style="color:blue"> Cas confirmés de Covid-19 dans le monde </span>

# In[3]:


df_cases = pd.read_csv("data/CONVENIENT_global_confirmed_cases.csv")


#  - ### 2.3 <span style="color:blue"> Décès liés au Covid-19 dans le monde </span>

# In[4]:


df_deaths = pd.read_csv("data/CONVENIENT_global_deaths.csv")


#  - ### 2.4 <span style="color:blue"> Population mondiale en 2020 </span>

# **Metadonnées open source disponibles sur** 
# https://population.un.org/wpp/Download/Standard/CSV/

# In[5]:


pop = pd.read_csv("data/population_by_country_2020.csv")


#  - ### 2.5 <span style="color:blue"> Nombre de morts dans le monde (Raw file) </span>

# **Ce fichier sera utilisé pour la country visualisation des nombres de morts**

# In[6]:


df_d = pd.read_csv('data/RAW_global_deaths.csv')


# # 3.<span style="color:red"> EDA & Preprocessing </span>

#  - ### 3.1 <span style="color:blue"> Extraction des données du continent Européen </span>

# In[7]:


df_cases.drop('United Kingdom',axis=1, inplace = True)
df_cases.drop('France',axis=1, inplace = True) 
df_cases.drop('Netherlands',axis=1, inplace = True)
df_cases.columns = [i.replace('United Kingdom.6','Gibraltar') for i in df_cases.columns]
df_cases.columns = [i.replace('United Kingdom.7','Isle of Man') for i in df_cases.columns]
df_cases.columns = [i.replace('United Kingdom.11','United Kingdom')for i in df_cases.columns]
df_cases.columns = [i.replace('France.11','France')for i in df_cases.columns]
df_cases.columns = [i.replace('Czechia','Czech republic')for i in df_cases.columns]
df_cases.columns = [i.replace('Netherlands.4','Netherlands')for i in df_cases.columns]

#---------#
df_deaths.drop('United Kingdom',axis=1, inplace = True)
df_deaths.drop('France',axis=1, inplace = True)
df_deaths.drop('Netherlands',axis=1, inplace = True)
df_deaths.columns = [i.replace('United Kingdom.6','Gibraltar') for i in df_deaths.columns]
df_deaths.columns = [i.replace('United Kingdom.7','Isle of Man') for i in df_deaths.columns]
df_deaths.columns = [i.replace('United Kingdom.11','United Kingdom')for i in df_deaths.columns]
df_deaths.columns = [i.replace('France.11','France')for i in df_deaths.columns]
df_deaths.columns = [i.replace('Czechia','Czech republic')for i in df_deaths.columns]
df_deaths.columns = [i.replace('Netherlands.4','Netherlands')for i in df_deaths.columns]


# In[8]:


#--------#Denmark Data aggregation
df_cases['Denmark'] = df_cases[['Denmark','Denmark.1','Denmark.2']].sum(axis=1)
df_deaths['Denmark'] = df_deaths[['Denmark','Denmark.1','Denmark.2']].sum(axis=1)


# In[9]:


#Dropping first line
df_cases.drop (0, inplace = True)
df_deaths.drop (0, inplace = True)


#  - ### 3.2 <span style="color:blue"> Indexation par format: datetime </span>

# In[10]:


#1st column rename
df_cases.rename(columns={"Country/Region": "Date"}, inplace= True)
df_deaths.rename(columns={"Country/Region": "Date"}, inplace= True)


# In[11]:


df_cases.head()


# In[12]:


#--------#Setting Datetime index
df_cases.set_index(df_cases["Date"], inplace = True,)
df_deaths.set_index(df_cases["Date"], inplace = True,)
df_cases.drop ("Date",  axis = 1, inplace = True)
df_deaths.drop ("Date",  axis = 1, inplace = True)

#--------#Datetime format index
df_cases.index = pd.to_datetime(df_cases.index)
df_deaths.index = pd.to_datetime(df_deaths.index)

#--------#Setting Index Frequency to "Day"
df_cases.index.freq = "D"
df_deaths.index.freq = "D"

#--------#Memory usage reduction
df_cases = df_cases.astype(np.float32)
df_deaths = df_deaths.astype(np.float32)


# In[13]:


#--------#List of analyzed countries
EUROPE = [ 'Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia',
 'Czech republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Gibraltar', 'Greece', 'Holy See', 'Hungary',
 'Iceland', 'Ireland', 'Isle of Man', 'Italy', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg',
 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland', 'Portugal',
 'Romania', 'Russia', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Ukraine',
 'United Kingdom']


# In[14]:


#--------#Update of the dataframes
df_cases= df_cases[EUROPE]
df_deaths=df_deaths[EUROPE]


#  - ### 3.3 <span style="color:blue"> Extraction de la population du continent Européen </span>

# In[15]:


#--------#European countries population on 2020
pop.rename(columns={"Country (or dependency)": "Country", "Population (2020)":"Population"}, inplace= True)
pop.set_index(pop["Country"], inplace = True,)
pop = pd.DataFrame(pop['Population'], columns=["Population"])
pop.index = [i.replace('Czech Republic (Czechia)','Czech republic')for i in pop.index]
pop = pop[pop.index.isin(EUROPE)]
pop.sort_index(ascending = True, inplace=True)
pop.head()


#  - ### 3.4 <span style="color:blue"> Arrangement du raw file Nombre de morts dans le monde </span>

# In[16]:


df_d.loc[df_d.loc[:,'Province/State'] == 'Isle of Man',['Country/Region']] = 'Isle of Man'
df_d.loc[df_d.loc[:,'Province/State'] == 'Gibraltar',['Country/Region']] = 'Gibraltar'
df_d.loc[df_d.loc[:,'Country/Region'] == 'Czechia',['Country/Region']] = 'Czech republic'
df_d = df_d[df_d.loc[:,'Country/Region'].isin(EUROPE) & (df_d.loc[:,'Province/State'].isna() | df_d.loc[:,'Province/State'].eq('Isle of Man') | df_d.loc[:,'Province/State'].eq('Gibraltar'))]
df_d = df_d[['Country/Region','Lat','Long','6/22/21']]
df_d.columns = ['Country','Lat','Long','Total']
df_d.set_index("Country", inplace=True)
df_d.head()


#  - ### 3.5 <span style="color:blue"> Stats descriptives </span>

# **Vérification des Nan**

# In[17]:


df_cases.isna().sum()


# In[18]:


df_deaths.isna().sum()


# In[19]:


df_d.isna().sum()


# **Count/Mean/STD/Min/25%/50%/75%/Max**

# In[20]:


df_cases.describe().T


# In[21]:


df_deaths.describe().T


#  - ### 3.6 <span style="color:blue"> Traitement des outliers </span>

# - **Les valeurs négatives dans les cas confirmés** sont dûes à des corrections de doublons antérieurs PCR/Antigénique sur la même personne dans la même journée. Plusieurs articles et communiqués officiels expliquent ces changements.
# 
# - **Les valeurs négatives des décès** sont dûes aux corrections ultérieures aux rapports d'autopsies qui confirment ou infirment la cause de la mort par le COV-19
# 
# 
# 
# https://www.santepubliquefrance.fr/les-actualites/2021/des-indicateurs-plus-precis-pour-le-suivi-des-cas-confirmes-de-covid-19

# <u>**Outliers des nombres de Cas**</u>

# In[22]:


negative_cases = df_cases.loc[:,(df_cases.lt(0).any())]


# In[23]:


for i in negative_cases.lt(0).sum().index:
    print(i)
    print(negative_cases[i][(negative_cases[[i]] < 0).all(1)])
   


# In[24]:


df_cases["Czech republic"].loc["2021-03-22"] = round(df_cases["Czech republic"].loc["2021-03-15":"2021-03-21"].mean(axis=0),0)
#-------#
df_cases["Denmark"].loc["2021-03-24"] = round(df_cases["Denmark"].loc["2021-03-17":"2021-03-23"].mean(axis=0),0)
#-------#
df_cases["Finland"].loc["2020-07-15":"2020-07-16"] = round(df_cases["Finland"].loc["2020-07-08":"2020-07-14"].mean(axis=0),0)
#-------#
df_cases["France"].loc["2020-04-04"] = round(df_cases["France"].loc["2020-04-01":"2020-04-30"].mean(axis=0),0)
df_cases["France"].loc["2020-04-07"] = round(df_cases["France"].loc["2020-04-01":"2020-04-30"].mean(axis=0),0)
df_cases["France"].loc["2020-04-23"] = round(df_cases["France"].loc["2020-04-01":"2020-04-30"].mean(axis=0),0)
df_cases["France"].loc["2020-04-29"] = round(df_cases["France"].loc["2020-04-01":"2020-04-30"].mean(axis=0),0)

df_cases["France"].loc["2020-05-24"] = round(df_cases["France"].loc["2020-05-17":"2020-05-23"].mean(axis=0),0)

df_cases["France"].loc["2020-06-02"] = round(df_cases["France"].loc["2020-06-01":"2020-06-30"].mean(axis=0),0)
df_cases["France"].loc["2020-06-03"] = round(df_cases["France"].loc["2020-06-01":"2020-06-30"].mean(axis=0),0)
df_cases["France"].loc["2020-06-28"] = round(df_cases["France"].loc["2020-06-01":"2020-06-30"].mean(axis=0),0)

df_cases["France"].loc["2020-11-04"] = round(df_cases["France"].loc["2020-11-01":"2020-11-30"].mean(axis=0),0)

df_cases["France"].loc["2021-02-04"] = round(df_cases["France"].loc["2021-01-28":"2021-02-03"].mean(axis=0),0)

df_cases["France"].loc["2021-04-03"] = round(df_cases["France"].loc["2021-04-01":"2021-04-30"].mean(axis=0),0)
df_cases["France"].loc["2021-04-07"] = round(df_cases["France"].loc["2021-04-01":"2021-04-30"].mean(axis=0),0)
df_cases["France"].loc["2021-04-09"] = round(df_cases["France"].loc["2021-04-01":"2021-04-30"].mean(axis=0),0)
df_cases["France"].loc["2021-04-10"] = round(df_cases["France"].loc["2021-04-01":"2021-04-30"].mean(axis=0),0)

df_cases["France"].loc["2021-05-20"] = round(df_cases["France"].loc["2021-02-01":"2021-04-30"].mean(axis=0),0)

df_cases["France"].loc["2021-06-21"] = round(df_cases["France"].loc["2021-06-14":"2021-06-20"].mean(axis=0),0)
#-------#
df_cases["Gibraltar"].loc["2020-05-13"] = round(df_cases["Gibraltar"].loc["2020-05-07":"2020-05-13"].median(axis=0),0)
df_cases["Gibraltar"].loc["2021-04-23"] = round(df_cases["Gibraltar"].loc["2021-04-16":"2021-04-22"].median(axis=0),0)
df_cases["Gibraltar"].loc["2021-05-24"] = round(df_cases["Gibraltar"].loc["2021-05-17":"2021-05-23"].median(axis=0),0)
#-------#
df_cases["Iceland"].loc["2021-02-08"] = round(df_cases["Iceland"].loc["2021-02-01":"2021-02-07"].mean(axis=0),0)
#-------#
df_cases["Italy"].loc["2020-06-19"] = round(df_cases["Italy"].loc["2020-06-12":"2020-06-18"].mean(axis=0),0)
#-------#
df_cases["Luxembourg"].loc["2020-08-28"] = round(df_cases["Luxembourg"].loc["2020-08-21":"2020-08-27"].mean(axis=0),0)
#-------#
df_cases["Malta"].loc["2020-08-16"] = round(df_cases["Malta"].loc["2020-08-09":"2020-08-15"].mean(axis=0),0)
#-------#
df_cases["Monaco"].loc["2020-09-02"] = round(df_cases["Monaco"].loc["2020-08-26":"2020-09-01"].mean(axis=0),0)
#-------#
df_cases["Portugal"].loc["2020-05-02"] = round(df_cases["Portugal"].loc["2020-08-26":"2020-09-01"].mean(axis=0),0)
#-------#
df_cases["San Marino"].loc["2020-05-10"] = round(df_cases["San Marino"].loc["2020-05-03":"2020-05-09"].mean(axis=0),0)
df_cases["San Marino"].loc["2020-09-05"] = round(df_cases["San Marino"].loc["2020-08-26":"2020-09-01"].mean(axis=0),0)
#-------#
df_cases["Spain"].loc["2020-04-24"] = round(df_cases["Spain"].loc["2020-04-17":"2020-04-23"].mean(axis=0),0)
df_cases["Spain"].loc["2020-05-25"] = round(df_cases["Spain"].loc["2020-05-18":"2020-05-24"].mean(axis=0),0)
df_cases["Spain"].loc["2021-03-02"] = round(df_cases["Spain"].loc["2021-02-01":"2021-02-28"].mean(axis=0),0)
#-------#
df_cases["United Kingdom"].loc["2021-04-09"] = round(df_cases["United Kingdom"].loc["2021-04-02":"2021-04-08"].mean(axis=0),0)
df_cases["United Kingdom"].loc["2021-05-18"] = round(df_cases["United Kingdom"].loc["2021-05-11":"2021-05-17"].mean(axis=0),0)


# In[25]:


df_cases.loc[:,(df_cases.lt(0).any())]


# <u>**Outliers des nombres de Morts**</u>

# In[26]:


negative_deaths = df_deaths.loc[:,(df_deaths.lt(0).any())]


# In[27]:


for i in negative_deaths.lt(0).sum().index:
    print(i)
    print(negative_deaths[i][(negative_deaths[[i]] < 0).all(1)])


# In[28]:


df_deaths["Austria"].loc["2020-07-21"] = round(df_deaths["Austria"].loc["2020-07-14":"2020-07-20"].mean(axis=0),0)
df_deaths["Austria"].loc["2020-10-11"] = round(df_deaths["Austria"].loc["2020-10-04":"2020-10-10"].mean(axis=0),0)
#-------#
df_deaths["Belgium"].loc["2020-08-26"] = round(df_deaths["Belgium"].loc["2020-08-19":"2020-08-25"].mean(axis=0),0)
#-------#
df_deaths["Bosnia and Herzegovina"].loc["2020-12-28"] = round(df_deaths["Bosnia and Herzegovina"].loc["2020-12-21":"2020-12-27"].mean(axis=0),0)
#-------#
df_deaths["Czech republic"].loc["2020-05-18"] = round(df_deaths["Czech republic"].loc["2020-05-11":"2020-05-17"].mean(axis=0),0)
df_deaths["Czech republic"].loc["2020-06-11"] = round(df_deaths["Czech republic"].loc["2020-06-04":"2020-06-10"].mean(axis=0),0)
df_deaths["Czech republic"].loc["2020-06-13"] = round(df_deaths["Czech republic"].loc["2020-06-06":"2020-06-12"].mean(axis=0),0)
df_deaths["Czech republic"].loc["2020-06-28"] = round(df_deaths["Czech republic"].loc["2020-06-21":"2020-06-27"].mean(axis=0),0)
df_deaths["Czech republic"].loc["2020-07-04"] = round(df_deaths["Czech republic"].loc["2020-06-28":"2020-07-03"].mean(axis=0),0)
df_deaths["Czech republic"].loc["2020-07-05"] = round(df_deaths["Czech republic"].loc["2020-06-29":"2020-07-04"].mean(axis=0),0)
df_deaths["Czech republic"].loc["2020-08-04"] = round(df_deaths["Czech republic"].loc["2020-07-28":"2020-08-03"].mean(axis=0),0)
df_deaths["Czech republic"].loc["2020-08-07"] = round(df_deaths["Czech republic"].loc["2020-07-31":"2020-08-06"].mean(axis=0),0)
#-------#
df_deaths["Denmark"].loc["2020-05-12"] = round(df_deaths["Denmark"].loc["2020-05-05":"2020-05-11"].mean(axis=0),0)
#-------#
df_deaths["Estonia"].loc["2020-08-02"] = round(df_deaths["Estonia"].loc["2020-07-26":"2020-08-01"].mean(axis=0),0)
#-------#
df_deaths["Finland"].loc["2020-04-06"] = round(df_deaths["Finland"].loc["2020-03-30":"2020-04-05"].mean(axis=0),0)
df_deaths["Finland"].loc["2020-06-01"] = round(df_deaths["Finland"].loc["2020-05-24":"2020-05-30"].mean(axis=0),0)
df_deaths["Finland"].loc["2020-07-15"] = round(df_deaths["Finland"].loc["2020-07-08":"2020-07-14"].mean(axis=0),0)
df_deaths["Finland"].loc["2020-09-30"] = round(df_deaths["Finland"].loc["2020-09-23":"2020-09-29"].mean(axis=0),0)
df_deaths["Finland"].loc["2020-10-23"] = round(df_deaths["Finland"].loc["2020-10-16":"2020-10-22"].mean(axis=0),0)
df_deaths["Finland"].loc["2021-05-18"] = round(df_deaths["Finland"].loc["2021-05-11":"2021-05-17"].mean(axis=0),0)
#-------#
df_deaths["France"].loc["2020-05-19"] = round(df_deaths["France"].loc["2020-05-12":"2020-05-18"].mean(axis=0),0)
df_deaths["France"].loc["2020-05-24"] = round(df_deaths["France"].loc["2020-05-17":"2020-05-23"].mean(axis=0),0)
df_deaths["France"].loc["2020-07-21"] = round(df_deaths["France"].loc["2020-07-14":"2020-07-20"].mean(axis=0),0)
df_deaths["France"].loc["2020-09-04"] = round(df_deaths["France"].loc["2020-08-29":"2020-09-03"].mean(axis=0),0)
df_deaths["France"].loc["2020-10-25"] = round(df_deaths["France"].loc["2020-10-18":"2020-10-24"].mean(axis=0),0)
df_deaths["France"].loc["2020-11-04"] = round(df_deaths["France"].loc["2020-10-28":"2020-11-03"].mean(axis=0),0)
df_deaths["France"].loc["2021-02-04"] = round(df_deaths["France"].loc["2021-01-28":"2021-02-03"].mean(axis=0),0)
df_deaths["France"].loc["2021-04-03"] = round(df_deaths["France"].loc["2021-03-27":"2021-04-02"].mean(axis=0),0)
#-------#
df_deaths["Germany"].loc["2020-04-11"] = round(df_deaths["Germany"].loc["2020-04-04":"2020-04-10"].mean(axis=0),0)
df_deaths["Germany"].loc["2020-07-06"] = round(df_deaths["Germany"].loc["2020-06-29":"2020-07-05"].mean(axis=0),0)
#-------#
df_deaths["Iceland"].loc["2020-03-16"] = round(df_deaths["Iceland"].loc["2020-03-09":"2020-03-15"].mean(axis=0),0)
df_deaths["Iceland"].loc["2020-03-20"] = round(df_deaths["Iceland"].loc["2020-03-13":"2020-03-19"].mean(axis=0),0)
#-------#
df_deaths["Ireland"].loc["2020-05-25"] = round(df_deaths["Ireland"].loc["2020-05-18":"2020-05-24"].mean(axis=0),0)
df_deaths["Ireland"].loc["2020-06-01"] = round(df_deaths["Ireland"].loc["2020-05-25":"2020-05-31"].mean(axis=0),0)
df_deaths["Ireland"].loc["2020-07-08"] = round(df_deaths["Ireland"].loc["2020-07-01":"2020-07-07"].mean(axis=0),0)
df_deaths["Ireland"].loc["2020-07-30"] = round(df_deaths["Ireland"].loc["2020-07-23":"2020-07-29"].mean(axis=0),0)
df_deaths["Ireland"].loc["2020-10-02"] = round(df_deaths["Ireland"].loc["2020-09-24":"2020-10-01"].mean(axis=0),0)
df_deaths["Ireland"].loc["2020-12-08"] = round(df_deaths["Ireland"].loc["2020-12-01":"2020-12-07"].mean(axis=0),0)
df_deaths["Ireland"].loc["2021-05-07"] = round(df_deaths["Ireland"].loc["2021-04-30":"2021-05-06"].mean(axis=0),0)
#-------#
df_deaths["Italy"].loc["2020-06-24"] = round(df_deaths["Italy"].loc["2020-06-17":"2020-06-23"].mean(axis=0),0)
#-------#
df_deaths["Luxembourg"].loc["2020-04-14"] = round(df_deaths["Luxembourg"].loc["2020-04-07":"2020-04-13"].mean(axis=0),0)
#-------#
df_deaths["Malta"].loc["2020-11-03"] = round(df_deaths["Malta"].loc["2020-10-27":"2020-11-02"].mean(axis=0),0)
#-------#
df_deaths["Monaco"].loc["2020-09-02"] = round(df_deaths["Monaco"].loc["2020-08-26":"2020-09-01"].mean(axis=0),0)
#-------#
df_deaths["Netherlands"].loc["2020-07-10"] = round(df_deaths["Netherlands"].loc["2020-07-02":"2020-07-09"].mean(axis=0),0)
df_deaths["Netherlands"].loc["2020-07-14"] = round(df_deaths["Netherlands"].loc["2020-07-07":"2020-07-13"].mean(axis=0),0)
df_deaths["Netherlands"].loc["2020-07-18"] = round(df_deaths["Netherlands"].loc["2020-07-11":"2020-07-17"].mean(axis=0),0)
df_deaths["Netherlands"].loc["2020-07-27"] = round(df_deaths["Netherlands"].loc["2020-07-10":"2020-07-26"].mean(axis=0),0)
df_deaths["Netherlands"].loc["2020-08-11"] = round(df_deaths["Netherlands"].loc["2020-08-04":"2020-08-10"].mean(axis=0),0)
#-------#
df_deaths["Norway"].loc["2021-06-07"] = round(df_deaths["Norway"].loc["2021-05-30":"2021-06-06"].mean(axis=0),0)
#-------#
df_deaths["Serbia"].loc["2020-03-26"] = round(df_deaths["Serbia"].loc["2020-03-19":"2020-03-25"].mean(axis=0),0)
#-------#
df_deaths["Slovakia"].loc["2020-03-22"] = round(df_deaths["Slovakia"].loc["2020-03-15":"2020-03-21"].mean(axis=0),0)
#-------#
df_deaths["Spain"].loc["2020-05-25"] = round(df_deaths["Spain"].loc["2020-05-18":"2020-05-24"].mean(axis=0),0)
df_deaths["Spain"].loc["2020-08-12"] = round(df_deaths["Spain"].loc["2020-08-05":"2020-08-11"].mean(axis=0),0)
#-------#
df_deaths["Sweden"].loc["2020-04-04"] = round(df_deaths["Sweden"].loc["2020-03-28":"2020-04-03"].mean(axis=0),0)
df_deaths["Sweden"].loc["2020-08-07"] = round(df_deaths["Sweden"].loc["2020-07-31":"2020-08-06"].mean(axis=0),0)
df_deaths["Sweden"].loc["2020-09-01"] = round(df_deaths["Sweden"].loc["2020-08-25":"2020-08-31"].mean(axis=0),0)
df_deaths["Sweden"].loc["2020-10-07"] = round(df_deaths["Sweden"].loc["2020-09-30":"2020-10-06"].mean(axis=0),0)
df_deaths["Sweden"].loc["2020-10-28"] = round(df_deaths["Sweden"].loc["2020-10-21":"2020-10-27"].mean(axis=0),0)
#-------#
df_deaths["Switzerland"].loc["2020-10-21"] = round(df_deaths["Switzerland"].loc["2020-10-14":"2020-10-20"].mean(axis=0),0)
df_deaths["Switzerland"].loc["2021-04-25"] = round(df_deaths["Switzerland"].loc["2021-04-18":"2021-04-24"].mean(axis=0),0)
df_deaths["Switzerland"].loc["2021-06-17"] = round(df_deaths["Switzerland"].loc["2021-06-10":"2021-06-16"].mean(axis=0),0)


# In[29]:


df_deaths.loc[:,(df_deaths.lt(0).any())]


#  - ### 3.7 <span style="color:blue"> Création des Features </span>

# <u>**Total des cas et des Morts liés au COV-19 en Europe**</u>

# In[30]:


#--------#Total cases & deaths dataframes creation
SumOfCases = df_cases.sum()
SumOfDeaths = df_deaths.sum()

SumProportion = pd.DataFrame((SumOfDeaths / SumOfCases)*100, columns=["Morts/Cas"])
SumProportion.sort_values(by="Morts/Cas",ascending=False, inplace = True)

SumProportion.replace([np.inf, -np.inf, np.nan], 0 , inplace=True)
SumProportion.sort_values(by="Morts/Cas",ascending=False, inplace = True)


# <u>**Case fatality rate (Morts/Cas)**</u>

# On peut divisier les pays en 3 groupes suivant le case fatality rate 
# - Groupe1 = **elevé** 
# - Groupe2 = **moyen**
# - Groupe3 = **bas** 

# In[31]:


SumProportion['Groups'] = pd.qcut(SumProportion["Morts/Cas"],3)


# In[32]:


SumProportion


# In[33]:


Groupe1 = SumProportion[:15]
Groupe2 = SumProportion[15:30]
Groupe3 = SumProportion[30:]


# In[34]:


Groupe1


# In[35]:


Groupe2


# In[36]:


Groupe3


# <u>**Préparation des données à modéliser**</u>

# In[37]:


df_cases_x = df_cases.copy()
df_cases_x["Total"] = df_cases_x.sum(1)


# In[38]:


df_deaths_x = df_deaths.copy()
df_deaths_x["Total"] = df_deaths_x.sum(1)


# <u>**Crude Mortality rate per 100K population**</u>

# In[39]:


df_d["Population"] = pop["Population"]


# In[40]:


df_d["Mortality 100K pop"] = np.round((df_d["Total"]/df_d["Population"])*100000,0)


# In[41]:


df_d.head()


# # 4.<span style="color:red"> Data Viz </span>

#  - ### 4.1 <span style="color:blue"> Highest Case fatality rate </span>

# In[42]:


data=Groupe1

fig = px.bar(data, x=Groupe1.index, y="Morts/Cas",
             hover_data=["Morts/Cas"], color="Morts/Cas",
             labels={'pop':'Ratio'}, height=400,title='Highest Case Fatality rate in Europe')
fig.update_layout(template='plotly_dark')
fig.show()


#  - ### 4.2 <span style="color:blue"> Moderated Case fatality rate</span>

# In[43]:


data=Groupe2

fig = px.bar(data, x=Groupe2.index, y="Morts/Cas",
             hover_data=["Morts/Cas"], color="Morts/Cas",
             labels={'pop':'Ratio'}, height=400,title='Moderated Case Fatality rate in Europe')
fig.update_layout(template='plotly_dark')
fig.show()


#  - ### 4.3 <span style="color:blue"> Lowest Case fatality rate</span>

# In[44]:


data=Groupe3

fig = px.bar(data, x=Groupe3.index, y="Morts/Cas",
             hover_data=["Morts/Cas"], color="Morts/Cas",
             labels={'pop':'Ratio'}, height=400,title='Lowest Case Fatality rate in Europe')
fig.update_layout(template='plotly_dark')
fig.show()


#  - ### 4.4 <span style="color:blue"> 10 Highest Crude death rate</span>

# In[45]:


top_10 = df_d.sort_values("Mortality 100K pop",ascending = False).head(10)
top_10.reset_index(inplace=True)
data=top_10

fig = px.bar(data, x=top_10['Country'], y="Mortality 100K pop",
             hover_data=['Country'], color="Mortality 100K pop",
             labels={'pop':'Taux brut de décès'}, height=400,title='Highest Crude death rate in Europe')
fig.update_layout(template='ggplot2')
fig.show()


#  - ### 4.5 <span style="color:blue"> Map of Total deaths per country</span>

# In[46]:


df_d.reset_index(inplace=True)
fig = px.scatter_geo(df_d, lat=df_d['Lat'], lon = df_d['Long'],
                     hover_name=df_d['Country'],
                     size= df_d['Total'],
                     color= df_d['Total'],
                     projection="natural earth",
                    fitbounds = "locations",
                    title= "Total deaths per country",
                    opacity = 0.7)
fig


#  - ### 4.6 <span style="color:blue"> Cases monthly rolling mean</span>

# In[47]:


cases_chart=alt.Chart(df_cases_x.reset_index()).mark_line(point=True).encode(
    x='Date', 
    y="Total", 
    tooltip=['Date',"Total"])

#Create Rolling mean. This centered rolling mean 
rolling_mean = alt.Chart(df_cases_x.reset_index()).mark_trail(
    color='red',
    size=1
).transform_window(
    rolling_mean='mean(Total)',
    frame=[-15,15] #Moving average interval
).encode(
    x='Date:T', #T encoding for time data
    y='rolling_mean:Q', #Q encoding for continuous real-valued quantity
    size='Total')

#Add zoom-in/out
scales = alt.selection_interval(bind='scales')

#Combine everything
(cases_chart + rolling_mean).properties(
    width=900, 
    title="European COV-19 cases & Monthly Rolling mean").add_selection(
    scales
)


#  - ### 4.7 <span style="color:blue"> Deaths monthly rolling mean</span>

# In[48]:


cases_chart=alt.Chart(df_deaths_x.reset_index()).mark_line(point=True).encode(
    x='Date', 
    y="Total", 
    tooltip=['Date',"Total"])

#Create Rolling mean. This centered rolling mean 
rolling_mean = alt.Chart(df_deaths_x.reset_index()).mark_trail(
    color='orange',
    size=1
).transform_window(
    rolling_mean='mean(Total)',
    frame=[-15,15] #Moving average interval
).encode(
    x='Date:T', #T encoding for time data
    y='rolling_mean:Q', #Q encoding for continuous real-valued quantity
    size='Total')

#Add zoom-in/out
scales = alt.selection_interval(bind='scales')

#Combine everything
(cases_chart + rolling_mean).properties(
    width=900, 
    title="European COV-19 deaths & Monthly Rolling mean").add_selection(
    scales
)


# # 5.<span style="color:red"> Tests des séries </span>

# **Nous allons utiliser la somme journalière des nombres de cas et des nombres de morts pour pouvoir ensuite essayer les différents modèles de prédiction**

# In[49]:


df_cases = df_cases.sum(1)


# In[50]:


df_deaths = df_deaths.sum(1)


#  - ### 5.1 <span style="color:blue"> Test de stationnarité</span>

# In[51]:


def kpss_test(series, **kw):
    statistic, p_value, n_lags, critical_values= kpss(series, **kw)
    print(f'KPSS statistic: {statistic}')
    print(f'p_value: {p_value}')
    print(f'num lags: {n_lags}')
    print(f'Critical values:')
    for key, value in critical_values.items():
        print(f' {key} : {value}')
    print(f'Result: the series is {"not " if p_value < 0.05 else ""}stationary')


# In[52]:


#Test de stationnarité
kpss_results = {}
print(f'df_cases')
kpss_results = kpss_test(df_cases)


# In[53]:


#Test de stationnarité
kpss_results = {}
print(f'df_deaths')
kpss_results = kpss_test(df_deaths)


#  - ### 5.2 <span style="color:blue"> Stationnarisation</span>

# **Une différence première sera appliquée sur les séries pour les stationnariser et afin d'avoir une moyenne, une variance et une fonction d'autocorrélation plus ou moins constantes dans le temps**

# In[54]:


cases_diff = df_cases.copy()
deaths_diff = df_deaths.copy()


# In[55]:


cases_diff = df_cases.diff().dropna()


# In[56]:


deaths_diff = df_deaths.diff().dropna()


# In[57]:


#Test de stationnarité
kpss_results = {}
kpss_results = kpss_test(cases_diff)


# In[58]:


#Test de stationnarité
kpss_results = {}
kpss_results = kpss_test(deaths_diff)


#  - ### 5.3 <span style="color:blue"> Test de normalité</span>

# In[59]:


#Test de normalité (Shapiro-Wilk)
#H0 : distribution normale
#HA : pas de distribution normale
p_value = 0.05
a,b= stats.shapiro(cases_diff)
print ("Statistiques", a, "p_value", b)
if b < p_value:
    print(f"L'hypothèse nulle H0 peut être rejetée et la série cases_diff ne présente pas de distribution normale")
else:
    print(f"L'hypothèse nulle H0 ne peut être rejetée et la série cases_diff présente une distribution normale")
        


# In[60]:


#Test de normalité (Shapiro-Wilk)
#H0 : distribution normale
#HA : pas de distribution normale
p_value = 0.05
a,b= stats.shapiro(deaths_diff)
print ("Statistiques", a, "p_value", b)
if b < p_value:
    print(f"L'hypothèse nulle H0 peut être rejetée et la série deaths_diff ne présente pas de distribution normale")
else:
    print(f"L'hypothèse nulle H0 ne peut être rejetée et la série deaths_diff présente une distribution normale")
        


# # 6.<span style="color:red"> Décomposition & Corrélations </span>

#  - ### 6.1 <span style="color:blue"> Décomposition des Timeseries</span>

# In[61]:


sns.set_style('darkgrid')
pd.plotting.register_matplotlib_converters()
# Default figure size
sns.mpl.rc('figure',figsize=(16, 6))
plt.rcParams.update({'figure.max_open_warning': 0})


# **Le process de décomposition des Time Series consiste à séparer les données en composantes qui sont** :
#  - Une tendance (Trend) qui démontre une potentielle augmentation et baisse de la moyenne
#  - Une saisonnalité qui représente un cycle récurrent dans les données
#  - Les résiduels aléatoires restants après suppression de la trend et la saisonnalité
#  
#  
#  **Le modèle additif est choisi du fait que l'amplitude de la saisonnalité est indépendante de la moyenne ainsi que la présence de valeurs négatives/zéros après stationnarisation** 
#  
#  
#  **Un modèle additif suggère que les composants sont ajoutés de façon linéaire comme suit**:
#  - y(t) = Level + Trend + Seasonality + Noise

# In[62]:


decomp_cases = seasonal_decompose(cases_diff,model="additive")  #Additive due to negative/zeros values
fig = decomp_cases.plot()


# **Interprétation**:
#   - La trend et les résiduels sont assez explicites et expliquent bien les variations observées dans les données.
#   - La saisonalité (7 jours) a été détectée automatiquement  

# In[63]:


decomp_cases.seasonal.head(16)


# In[64]:


decomp_deaths = seasonal_decompose(df_deaths,model="additive") 
fig = decomp_deaths.plot()


# In[65]:


decomp_deaths.seasonal.head(16)


# **Interprétation**:
#   - La trend est beaucoup plus lisse ce qui est tout à fait normal au vue des données observées.
#   - La saisonalité (7 jours) a été détectée automatiquement 
#   - Les résiduels sont beaucoup plus importants

#  - ### 6.2 <span style="color:blue"> ACF: Autocorrelation plots</span>

# **Informations pouvant être extraite des plots ACF-PACF**.
# 
# - **Modèle AR** : 
#     1.  **ACF** --> Au fur et à mesure, après un certain point, il n'y a plus de relation.
#     2. **PACF** --> Après un certain nombre de lag, tout à coup la relation temporelle n'existe plus.
# - **Modèle MA** : 
#     1. **ACF** --> Après un certain nombre de lag, tout à coup la relation temporelle n'existe plus.
#     2. **PACF** --> Au fur et à mesure, après un certain point, il n'y a plus de relation.

# - **Analyse visuelle des autocorrelations et leurs significativité**

# In[66]:


plot_acf(cases_diff, lags = 100, title='Autocorrelation Nombre de Cas')
plt.show()

plot_acf(deaths_diff, lags = 100, title='Autocorrelation Nombre de Morts')
plt.show()


# **Le plot ACF est une representation des coéfficients de corrélation entre une Time series et ses valeurs antérieures.**
# 
# Dans les deux séries:
#  - **Les lags démontrent une tendance sinusoidale décroissante qui alterne entre corrélations positive et négative ce qui prévoit un processus AR+MA**. 
#  - **Les lags positifs récurrents chaque 7 jours correspondent à la période de saisonnalité**.
#  - **Terme autorégressif d'ordre supérieur dans les données**.

#  - ### 6.3 <span style="color:blue"> PACF: Partial autocorrelation plots</span>

# - **Analyse visuelle de la relation directe entre Y(t) et Y(t-1)**

# In[67]:


plot_pacf(cases_diff, lags = 100,title='Partial autocorrelation Nombre de Cas')
plt.show()

plot_pacf(deaths_diff, lags = 100, title="Partial autocorrelation Nombre de morts")
plt.show()


# **Le plot PACF représente la fonction d'autocorrélation partielle qui explique la corrélation partielle entre la série temporelle et ses propres lags avec une régression linéaire qui prédit y(t) à partir des valeurs antérieures y(t-1), y(t-2), y(t-3)...**
# 
#  - Pour les deux séries, on observe une baisse significative de l'importance des lags au bout de 9 jours pour les cas et 8 jours pour les morts
# 

#  - ### 6.4 <span style="color:blue"> Train & Test splits</span>

# In[68]:



train_cases, test_cases= np.split(cases_diff, [int(.80 *len(cases_diff))])
train_deaths, test_deaths= np.split(deaths_diff, [int(.80 *len(deaths_diff))])


# In[69]:


fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True)

train_cases.plot(ax=ax[0], grid=True, title="Train", color="blue",legend=None)
test_cases.plot(ax=ax[1], grid=True, title="Test", color="red", legend=None)

ax[0].set(xlabel=None)
ax[1].set(xlabel=None)
 
plt.show()


# In[70]:


fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True)

train_deaths.plot(ax=ax[0], grid=True, title="Train", color="orange",legend=None)
test_deaths.plot(ax=ax[1], grid=True, title="Test", color="green", legend=None)

ax[0].set(xlabel=None)
ax[1].set(xlabel=None)
 
plt.show()


# # 7.<span style="color:red"> Modélisation </span>

#  - ### 7.1 <span style="color:blue"> Random Walk</span>

# In[71]:


'''
Generate a Random Walk process.
: using the parameters of the Standard and Poor Weekly Data observations (we will use Adj Close)
    :param y0: num - observation #1 
    :param n: num - total number of observations
    :param ymin: num - limit
    :param ymax: num - limit
'''
def generate_rw(y0, n, sigma, ymin=None, ymax=None):
    random.seed(15)
    rw = [y0]
    for t in range(1, n):
        yt = rw[t-1] + np.random.normal(0,sigma)
        if (ymax is not None) and (yt > ymax):
            yt = rw[t-1] - abs(np.random.normal(0,sigma))
        elif (ymin is not None) and (yt < ymin):
            yt = rw[t-1] + abs(np.random.normal(0,sigma))
        rw.append(yt)
    return rw


# In[72]:


def simulate_rw(train, test):
    ## simulate train
    #diff_ts = train - train.shift(1)
    rw = generate_rw(y0=train[0], n=len(train), sigma=train.std(), ymin=train.min(), ymax=train.max())
    dtf_train = train.to_frame(name="data").merge(pd.DataFrame(rw, index=train.index, columns=["model"]), how='left', left_index=True, right_index=True)
    
    ## test
    rw = generate_rw(y0=test[0], n=len(test), 
                           sigma=train.std(), ymin=train.min(), 
                           ymax=train.max())
    dtf_test = test.to_frame(name="data").merge(
                 pd.DataFrame(rw, index=test.index, 
                              columns=["forecast"]), how='left', 
                              left_index=True, right_index=True)
    ## evaluate
    dtf = dtf_train.append(dtf_test)
    #dtf = utils_evaluate_forecast(dtf, figsize=figsize, 
                                  #title="Random Walk Simulation")
    return dtf


# In[73]:


result_cases = simulate_rw(train_cases, test_cases)
result_cases


# In[74]:


plt.plot(result_cases)
plt.legend(('Data', 'Model', 'Forecast'))
plt.show()


# In[75]:


result_deaths = simulate_rw(train_deaths, test_deaths)
result_deaths


# In[76]:


plt.plot(result_deaths)
plt.legend(('Data', 'Model', 'Forecast'))
plt.show()


#  - #### 7.1.a <span style="color:green"> RMSE : Random Walk</span>

# In[77]:


## residuals
result_cases["residuals"] = result_cases["data"] - result_cases["model"]
result_cases["error"] = result_cases["data"] - result_cases["forecast"]
result_cases["error_pct"] = result_cases["error"] / result_cases["data"]
        
## kpi
residuals_mean = result_cases["residuals"].mean()
residuals_std = result_cases["residuals"].std()
error_mean = result_cases["error"].mean()
error_std = result_cases["error"].std()
mae = result_cases["error"].apply(lambda x: np.abs(x)).mean()
mape = result_cases["error_pct"].apply(lambda x: np.abs(x)).mean()  
mse_rw = result_cases["error"].apply(lambda x: x**2).mean()
rmse_rw_cases = np.sqrt(mse_rw)  #root mean squared error


# In[78]:


## residuals
result_deaths["residuals"] = result_deaths["data"] - result_deaths["model"]
result_deaths["error"] = result_deaths["data"] - result_deaths["forecast"]
result_deaths["error_pct"] = result_deaths["error"] / result_deaths["data"]
        
## kpi
residuals_mean = result_deaths["residuals"].mean()
residuals_std = result_deaths["residuals"].std()
error_mean = result_deaths["error"].mean()
error_std = result_deaths["error"].std()
mae = result_deaths["error"].apply(lambda x: np.abs(x)).mean()
mape = result_deaths["error_pct"].apply(lambda x: np.abs(x)).mean()  
mse_rw_deaths = result_deaths["error"].apply(lambda x: x**2).mean()
rmse_rw_deaths = np.sqrt(mse_rw_deaths)  #root mean squared error


# In[79]:


rmse_rw_cases


# In[80]:


rmse_rw_deaths


# **Les RMSE du random walk sont très importantes**

#  - ### 7.2 <span style="color:blue"> AutoReg model sans Saisonalité</span>

# In[81]:


cases_diff = pd.DataFrame(cases_diff, index = cases_diff.index, columns = ["Total"])


# In[82]:


deaths_diff = pd.DataFrame(deaths_diff, index = cases_diff.index, columns = ["Total"])


#  - #### 7.2.a <span style="color:green"> Lag plots</span>

# In[83]:


lag_plot(cases_diff["Total"])
plt.title("Lag plot t-1 Nombre de cas")
plt.show();

lag_plot(deaths_diff["Total"])
plt.title("Lag plot t-1 Nombre de morts")
plt.show()


# **Concentration des points autour de zéro**
# 
# **Les outliers sont nombreux**

#  - #### 7.2.b <span style="color:green"> Choix des lags</span>

# In[84]:


mod = ar_select_order(cases_diff, maxlag=30,old_names = False)
mod.ar_lags


# In[85]:


mod = ar_select_order(deaths_diff, maxlag=50,old_names = False)
mod.ar_lags


#  - #### 7.2.c <span style="color:green"> Entraînement de la série: nombre de Cas</span>

# In[86]:


mod_cases = AutoReg(cases_diff, lags =[1, 2, 3, 4, 5, 6, 7, 8, 9],old_names = False)
res_cases = mod_cases.fit()
print(res_cases.summary())


#  - #### 7.2.d <span style="color:green"> Interprétation</span>

# 
# - **La méthode d'estimation : conditional maximum likelihood** 
# - **L'AIC (akaike information criteria)détermine la qualité du modèle autoregressif appliqué en prenant compte du maximum likelihood et des nombres de paramètres**.
# 
#   AIC = ${2k-2\ln({\hat {L}})}$
# 
# **Plus l'AIC est petit plus le modèle est performant**.
# 
# - **Les lags n° 1 & 7 sont les plus impactants avec une faible p-value et un coefficient relativement conséquent**
# - **La P|z| doit être inférieure à 5% pourque les coefficients soient significatifs ce qui est le cas de nos lags à part le L6**.
# - **Les 6 premiers lags ont un impact négatif sur les valeurs et les 3 derniers ont un impact positif sur les valeurs**.

#  - #### 7.2.e <span style="color:green"> Entraînement sur la série: nombre de Morts</span>

# In[87]:


mod_deaths = AutoReg(deaths_diff["Total"], lags =[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14],old_names = False)
res_deaths = mod_deaths.fit()
print(res_deaths.summary())


#  - #### 7.2.f <span style="color:green"> Interprétation</span>

# 
# - **Les lags n° 2 & 8 sont les plus impactants avec une faible p-value et un coefficient relativement conséquent**
# - **La P|z| doit être inférieure à 5% pourque les coefficients soient significatifs ce qui est le cas de nos lags à part le L6**.
# - **Les 5 premiers lags ont un impact négatif sur les valeurs et les restes ont un impact positif sur les valeurs**.

#  - #### 7.2.g <span style="color:green"> Prédictions</span>

# In[88]:


#1 = Start, 550 = End (517 observations sont déjà dans notre Timeseries)
fig = res_cases.plot_predict(450,550)
fig = res_deaths.plot_predict(450,550)


# **Les prédictions des nombres de Cas tendent à une linéarisation vers 0 avec des amplitudes très faibles**
# 
# **Les prédictions des nombres de morts montrent une très légère tendance à la baisse des amplitudes en gardant une moyenne constante dans le temps**

#  - #### 7.2.h <span style="color:green"> Models diagnostics</span>

# In[89]:


fig = res_cases.plot_diagnostics(lags=30)


# **interprétation des résultats**
# 
#  - A. Standardized residuals
# 
# Le standardized residual est une mesure de la force de la différence entre les valeurs observées et attendues. C'est une mesure de l'importance des cellules par rapport à la valeur du chi-square. 
# Pour la série des nombres de cas, les cellules qui contribuent le plus aux vraies valeurs et qui sont supérieurs à 3 correspondent aux vagues de contaminations observées en **Avril 2020** , **Octobre 2020** , **Janvier 2021**  et **Avril 2021**.
# Notons aussi la présence des outliers surtout en **Octobre 2020** qui correspont au plus fort de la pandémie.
# 
# 
#  - B. Histogram
# L'histogramme KDE détermine la meilleure distribution possible de la donnée.
# 
#  - C. QQ Plot
# Le Q-Q plot, or quantile-quantile plot, est une représentation graphique des quantiles, pour la série nombres de cas, on observe des queues droite & gauche assez épaisse avec des outliers  (-5 , 6)

# In[90]:


fig = res_deaths.plot_diagnostics(lags=30)


# **interprétation des résultats**
# 
#  - A. Standardized residuals
# 
# Pour la série des nombres de morts, les cellules qui contribuent le plus aux vraies valeurs et qui sont supérieurs à 3 correspondent aux morts suite aux vagues de contaminations  observées en **Avril 2020**, **Juin 2020** , **Novembre 2020** , **Janvier 2021**  et **Avril 2021**.
# Notons aussi la présence des outliers surtout en **Novembre 2020** qui vient juste après le pic de pandémie en **Octobre2020**.
# 
# 
#  - B. Histogram
# L'histogramme KDE détermine la meilleure distribution possible de la donnée.
# 
#  - C. QQ Plot
# Le Q-Q plot, or quantile-quantile plot, est une représentation graphique des quantiles, pour la série nombres de morts, on observe une queue droite épaisse et avec des outliers  (-5 , 6)

#  - ### 7.2 (bis) <span style="color:blue"> AutoReg model avec Saisonalité</span>

#  - #### 7.2.a <span style="color:green"> Choix des lags et Entraînement de la série: nombre de Cas</span>

# In[91]:


sel_C = ar_select_order(cases_diff,30, seasonal=True)
sel_C.ar_lags
res_C = sel_C.model.fit()
print(res_C.summary())


#  - #### 7.2.b <span style="color:green"> Interprétation</span>

# 
# - **La saisonalité.2 est la plus impactante avec une p_value<5% et un coefficient négatif assez important**
# - **l'AIC ne s'est pas vraiment amélioré par rapport à un AutoReg model sans contrôle de saisonnalité**.

#  - #### 7.2.c <span style="color:green"> Choix des lags et Entraînement de la série: nombre de morts</span>

# In[92]:


sel_D = ar_select_order(deaths_diff, 30, seasonal=True)
sel_D.ar_lags
res_D = sel_D.model.fit()
print(res_D.summary())


#  - #### 7.2.d <span style="color:green"> Interprétation</span>

# 
# - **Les deux premières saisonalités sont les plus impactantes avec une p_value<5% et un coefficient négatif assez important**
# - **l'AIC ne s'est pas vraiment amélioré par rapport à un AutoReg model sans contrôle de saisonnalité**.

#  - #### 7.2.e <span style="color:green"> Prédictions</span>

# In[93]:


#1 = Start, 550 = End (517 observations sont déjà dans notre Timeseries)
fig = res_C.plot_predict(450,550)
fig = res_D.plot_predict(450,550)


# **Les prédictions des nombres de Cas gardent une amplitude légèrement croissante ce qui prévoit une croissance des nombres de cas**
# 
# **Les prédictions des nombres de morts montrent une tendance à la hausse des amplitudes ce qui prévoit une augmentation des nombres de morts dans le temps**

#  - #### 7.2.f <span style="color:green"> Models diagnostics</span>

# In[94]:


fig = res_C.plot_diagnostics(lags=30)


# **Les diagnostics sont sensiblement similaires à l'AutoReg sans saisonnalité**

# In[95]:


fig = res_D.plot_diagnostics(lags=30)


# **La distribution des Quantiles des nombres de morts est plus aplatie à gauche et moins épaisse à droite**

#  - ### 7.3 <span style="color:blue"> ARMA model</span>

#  - #### 7.3.a <span style="color:green"> AutoArima & Interprétation sur la série: Nombre de Cas</span>

# In[96]:


cases_model = auto_arima(cases_diff)
cases_model.summary()


# **Le modèle préconisé par AutoArima est SARIMAX (5,0,4) avec un AutoReg d'ordre 5, pas d'intégration puisque la série est stationnaire et une Moving Average d'ordre 4 avec une composante saisonnalité**

#  - #### 7.3.b <span style="color:green"> AutoArima & Interprétaion sur la série: Nombre de Morts</span>

# In[97]:


deaths_model = auto_arima(deaths_diff)
deaths_model.summary()


# **Le modèle préconisé par AutoArima est SARIMAX (2,0,2) avec un AutoReg d'ordre 2, pas d'intégration puisque la série est stationnaire et une Moving Average d'ordre 2 avec une composante saisonnalité**

#  - #### 7.3.c <span style="color:green"> SARIMAX sur la série: Nombre de Cas</span>

# In[98]:


cases_model = sm.tsa.statespace.SARIMAX(train_cases,order=(5,0,4), seasonal_order=(1,0,1,7))


# In[99]:


results_cases = cases_model.fit()
results_cases.summary()


# - **Les lags 4 & 5 sont les plus significatifs statistiquement avec des p_value<5% et des coefficients négatifs**
# - **La saisonnalité 7jours impacte positivement les valeurs**
# - **L'AIC même si il est amélioré par rapport à l'estimation de l'autoarima reste très elevé**

# In[100]:


results_cases.resid.plot();


# - **Les résiduels démontrent les pics épidémiques les plus importants et leurs impacts sur les valeurs de la série temporelle**

#  - #### 7.3.d <span style="color:green"> SARIMAX sur la série: Nombre de Morts</span>

# In[101]:


deaths_model = sm.tsa.statespace.SARIMAX(train_deaths,order=(2,0,2), seasonal_order=(1,0,1,7))


# In[102]:


results_deaths = deaths_model.fit()
results_deaths.summary()


# - **Le lag n°2 est le plus significatif avec un coefficient positif relativement important**
# - **La saisonnalité 7jours impacte positivement les valeurs**
# - **La Moving Average du Lag2 impacte négativement les valeurs avec un coefficient négatif important**
# - **L'AIC même si il est amélioré par rapport à l'estimation de l'autoarima reste très elevé**

# In[103]:


results_deaths.resid.plot();


# - **Les résiduels démontrent les pics de mortalités qui suivent les pics épidémiques les plus importants et leurs impacts sur les valeurs de la série temporelle**

#  - #### 7.3.e <span style="color:green"> Prédictions ARMA</span>

# In[104]:


def forecast_to_df(model, steps=7):
    forecast = model.get_forecast(steps=steps)
    pred_df = forecast.conf_int()
    pred_df['pred'] = forecast.predicted_mean
    pred_df.columns = ['lower', 'upper', 'pred']
    return pred_df


# In[105]:


len(test_cases)


# In[106]:


pred_df = forecast_to_df(results_cases, steps = len(test_cases))


# In[107]:


series_cases = test_cases - pred_df["pred"]
mse_arma_cases = series_cases.apply(lambda x: x**2).mean()
rmse_arma_cases = np.sqrt(mse_arma_cases)  #root mean squared error


# In[108]:


pred = pred_df['pred']


# In[109]:


def plot_train_test_pred(train,test,pred_df):
    fig,ax = plt.subplots(figsize=(12,7))
    kws = dict(marker='o')
    
    ax.plot(train,label='Train',**kws)
    ax.plot(test,label='Test',**kws)
    ax.plot(pred_df['pred'],label='Prediction',ls='--',linewidth=3)

    ax.fill_between(x=pred_df.index,y1=pred_df['lower'],y2=pred_df['upper'],alpha=0.3)
    ax.set_title('Model Validation', fontsize=22)
    ax.legend(loc='upper left')
    fig.tight_layout()
    return fig,ax


# In[110]:


plot_train_test_pred(train_cases,test_cases,pred_df)


# In[111]:


pred_deaths = forecast_to_df(results_deaths, steps = len(test_deaths))


# In[112]:


pred = pred_deaths['pred']


# In[113]:


plot_train_test_pred(train_deaths,test_deaths,pred_deaths);


# In[114]:


series_deaths = test_deaths - pred_deaths['pred']
mse_arma_deaths = series_deaths.apply(lambda x: x**2).mean()
rmse_arma_deaths = np.sqrt(mse_arma_deaths)


#  - #### 7.3.f <span style="color:green"> RMSE ARMA vs RMSE RW</span>

# In[115]:


print(f"RMSE Cases avec Random walk : {rmse_rw_cases}")
print("")
print(f"RMSE Cases avec Arma : {rmse_arma_cases}")
print("---------")
print(f"RMSE Deaths avec Random walk : {rmse_rw_deaths}")
print("")
print(f"RMSE Deaths avec Arma : {rmse_arma_deaths}")
print("")


# **On dénote une forte amélioration de la RMSE pour les deux séries avec un modèle de Random Walk et un modèle ARMA**

#  - ### 7.4 <span style="color:blue"> XGBoost model</span>

# **Credits:**  https://machinelearningmastery.com/xgboost-for-time-series-forecasting/

#  - #### 7.4.a <span style="color:green"> Suppression des Outliers avec IQR</span>

# In[116]:


Q1_cases = df_cases_x.quantile(0.25)
Q3_cases = df_cases_x.quantile(0.75)

# Then we define the interquantile range as the difference of the two.

IQR_cases = Q3_cases - Q1_cases
print(IQR_cases)


# In[117]:


Q1_deaths = df_deaths_x.quantile(0.25)
Q3_deaths = df_deaths_x.quantile(0.75)

# Then we define the interquantile range as the difference of the two.

IQR_deaths = Q3_deaths - Q1_deaths
print(IQR_deaths)


# In[118]:


cases_out =  df_cases_x[~((df_cases_x < (Q1_cases - 1.5 * IQR_cases)) |(df_cases_x > (Q3_cases + 1.5 * IQR_cases)))]


# In[119]:


cases_out.shape #Toutes les données ont été conservées puisque entrant dans l'intervalle Q1/Q3


# In[120]:


deaths_out =  df_deaths_x[~((df_deaths_x < (Q1_deaths - 1.5 * IQR_deaths)) |(df_deaths_x > (Q3_deaths + 1.5 * IQR_deaths)))]


# In[121]:


deaths_out.shape #Toutes les données ont été conservées puisque entrant dans l'intervalle Q1/Q3


# **En se basant sur les Case fatality rate, on utilisera les features précédèmment créées afin de grouper les pays en 3 groupes qui seront utilisées comme variables endogènes afin de prédire y: Total des Cas & y: Total des morts**

# In[122]:


G1 = list(Groupe1.index)
G2 = list(Groupe2.index)
G3 = list(Groupe3.index)


# In[123]:


X1_cases = df_cases_x[G1]
X2_cases = df_cases_x[G2]
X3_cases = df_cases_x[G3]


# In[124]:


X1_cases["Total"] = X1_cases.sum(axis=1)
X2_cases["Total"] = X2_cases.sum(axis=1)
X3_cases["Total"] = X3_cases.sum(axis=1)


# In[125]:


Y_cases =df_cases_x["Total"]


# In[126]:


X1_deaths = df_deaths_x[G1]
X2_deaths = df_deaths_x[G2]
X3_deaths = df_deaths_x[G3]


# In[127]:


X1_deaths["Total"] = X1_deaths.sum(axis=1)
X2_deaths["Total"] = X2_deaths.sum(axis=1)
X3_deaths["Total"] = X3_deaths.sum(axis=1)


# In[128]:


Y_deaths =df_deaths_x["Total"]


#  - #### 7.4.b <span style="color:green"> Stationnarisation</span>

# In[129]:


X1_cases = X1_cases.diff().dropna()
X2_cases = X2_cases.diff().dropna()
X3_cases = X3_cases.diff().dropna()
Y_cases = Y_cases.diff().dropna()
X1_deaths = X1_deaths.diff().dropna()
X2_deaths = X2_deaths.diff().dropna()
X3_deaths = X3_deaths.diff().dropna()
Y_deaths = Y_deaths.diff().dropna()


# In[130]:


cases = pd.concat([X1_cases["Total"], X2_cases["Total"],X3_cases["Total"],Y_cases], axis=1, keys = ["X1","X2","X3","Total"])


# In[131]:


deaths = pd.concat([X1_deaths["Total"], X2_deaths["Total"],X3_deaths["Total"],Y_deaths], axis=1, keys = ["X1","X2","X3","Total"])


#  - #### 7.4.c <span style="color:green"> Data transformation for Supervised learning dataset</span>

# **La fonction series_to_supervised()**:
# 
# Cette fonction prend une série temporelle univariée ou multivariée et la présente comme un ensemble de données d'apprentissage supervisé.
# 
# elle prendra 4 arguments:
# 
#  - data: Série temporelle sous forme de liste ou de Numpy Array 2D.
#  - n_in: Nombre d'observations lagguées (X). comprise entre [1..len(data)].
#  - n_out: Nombre d'observations output (y). comprise entre [0..len(data)-1].
#  - dropnan: suppression ou non des lignes avec NaN.
#  
# La fonction returnera un DataFrame pour un apprentissage supervisé.

# In[132]:


# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[0]
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


#  - #### 7.4.d <span style="color:green"> Data split</span>

# In[133]:


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]


#  - #### 7.4.e <span style="color:green"> XGB Regressor fit & Predict</span>

# In[134]:


# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX):
    # transform list into array
    train = np.asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(np.asarray([testX]))
    return yhat[0]


# In[135]:


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat = xgboost_forecast(history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        #print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    # estimate prediction error
    rmse = np.sqrt((mean_absolute_error(test[:, -1], predictions))**2)
    return rmse, test[:, -1], predictions
 


# In[136]:


# transform the time series data into supervised learning
data_cases = series_to_supervised(cases["Total"], n_in=30) #Shift de 30 jours
# evaluate
rmsexgb_cases, y, yhat = walk_forward_validation(data_cases, 15) #Prédiction de 15 jours

print('RMSE: %.3f' % rmsexgb_cases)
# plot expected vs preducted
plt.plot(y, label='Real data')
plt.plot(yhat, label='Predicted')
plt.title("Prédiction des nombres de Cas")
plt.legend()
plt.show()


# In[137]:


# transform the time series data into supervised learning
data_deaths = series_to_supervised(deaths["Total"], n_in=5) #Shift de 5 jours
# evaluate
rmsexgb_deaths, y, yhat = walk_forward_validation(data_deaths, 15) #Prédiction de 15 jours

print('RMSE: %.3f' % rmsexgb_deaths)
# plot expected vs preducted
plt.plot(y, label='Real data')
plt.plot(yhat, label='Predicted')
plt.title("Prédiction des nombres de Cas")
plt.legend()
plt.show()


#  - #### 7.4.f <span style="color:green"> RandomWalk vs ARMA vs XGBOOST</span>

# In[138]:


print(f"RMSE Cases avec Random walk : {rmse_rw_cases}")
print("")
print(f"RMSE Cases avec Arma : {rmse_arma_cases}")
print("")
print(f"RMSE Cases avec XGBoost : {rmsexgb_cases}")
print("---------")
print(f"RMSE Deaths avec Random walk : {rmse_rw_deaths}")
print("")
print(f"RMSE Deaths avec Arma : {rmse_arma_deaths}")
print("")
print(f"RMSE Deaths avec XGBoost : {rmsexgb_deaths}")


# # 8.<span style="color:red"> Amélioration des modèles </span>

# **Dans le cadre d'une potentielle amélioration pouvant être apportée, on a appliqué une approche expérimentale après analyse des données présentes**
# 
# **Vu que les premiers mois des datasets présentaient de faibles valeurs comparées à celles de du Q4 de 2020 et l'année 2021, on a tronqué la série jusqu'au 1er Août 2020**

#  - ### 8.1 <span style="color:blue"> Préparation des séries</span>

# In[139]:


partie = 190
Cases = cases_diff.iloc[partie:]
Deaths =deaths_diff.iloc[partie:]


# In[140]:


train_size = 0.8
split_idx_C_EX = round(len(Cases)* train_size)
split_idx_C_EX

split_idx_D_EX = round(len(Deaths)* train_size)
split_idx_D_EX
# Split
train_C_EX = Cases.iloc[:split_idx_C_EX]
test_C_EX = Cases.iloc[split_idx_C_EX:]

train_D_EX = Deaths.iloc[:split_idx_D_EX]
test_D_EX = Deaths.iloc[split_idx_D_EX:]


#  - ### 8.2 <span style="color:blue"> Seasonal decompose</span>

# In[141]:


result_C_EX = seasonal_decompose(Cases, model='additive')
result_C_EX.plot();


# - **La saisonalité est detectée automatiquement (7jours)**
# - **Les résiduels les plus importants correspondent aux pics épidémiologiques surtout Novembre 2020)**

# In[142]:


result_D_EX = seasonal_decompose(Deaths, model='additive')
result_D_EX.plot();


# - **La saisonalité est detectée automatiquement (7jours)**
# - **Les résiduels les plus importants correspondent avec un légèr décallage aux pics épidémiologiques surtout Mi-Janvier 2021 avec les nouveaux variants du virus)**

#  - ### 8.3 <span style="color:blue"> ACF & PACF</span>

# In[143]:


fig, axes = plt.subplots(nrows=1 ,ncols=2)

plot_acf(Cases,ax=axes[0], title="Cases Autocorrelation");
plot_acf(Deaths,ax=axes[1], title="Deaths Autocorrelation");


# Dans les deux séries:
#  - **Les lags démontrent une tendance en vague décroissant qui alterne entre corrélations positive et négative**.
#  - **Les lags positifs récurrents chaque 7 jours correspondent à la période de saisonnalité**.
#  - **Terme autorégressif d'ordre supérieur dans les données**.

# In[144]:


fig, axes = plt.subplots(nrows=1 ,ncols=2)

plot_pacf(Cases,ax=axes[0], title="Cases Partial Autocorrelation");
plot_pacf(Deaths,ax=axes[1], title="Deaths Partial Autocorrelation");


#  - **Pour les deux séries, on observe une baisse significative de l'importance des lags au bout de 9 jours pour les cas et 11 jours pour les morts**
# 

#  - ### 8.3 <span style="color:blue"> Random Walk</span>

#  - #### 8.3.a <span style="color:green"> Préparation des séries et split Train/Test</span>

# In[145]:


#Conversion d'un DF vers un pandas Series
Cases = Cases.squeeze()
Deaths = Deaths.squeeze()


# In[146]:


train_C_EX, test_C_EX= np.split(Cases, [int(.80 *len(Cases))])
train_D_EX, test_D_EX= np.split(Deaths, [int(.80 *len(Deaths))])


#  - #### 8.3.b <span style="color:green"> Application du RW et visualisation des prédictions</span>

# In[147]:


result_C_EX = simulate_rw(train_C_EX, test_C_EX)
result_D_EX = simulate_rw(train_D_EX, test_D_EX)


# In[148]:


plt.plot(result_C_EX)
plt.legend(('Data', 'Model', 'Forecast'))
plt.title('RW Cases')
plt.show()


# In[149]:


plt.plot(result_D_EX)
plt.legend(('Data', 'Model', 'Forecast'))
plt.title('RW Deaths')
plt.show()


#  - #### 8.3.c <span style="color:green"> RMSE</span>

# In[150]:


## residuals
result_C_EX["residuals"] = result_C_EX["data"] - result_C_EX["model"]
result_C_EX["error"] = result_C_EX["data"] - result_C_EX["forecast"]
result_C_EX["error_pct"] = result_C_EX["error"] / result_C_EX["data"]
        
## kpi
residuals_mean_C_EX = result_C_EX["residuals"].mean()
residuals_std_C_EX = result_C_EX["residuals"].std()
error_mean_C_EX = result_C_EX["error"].mean()
error_std_C_EX = result_C_EX["error"].std()
mse_C_EX = result_C_EX["error"].apply(lambda x: np.abs(x)).mean()
mape_C_EX = result_C_EX["error_pct"].apply(lambda x: np.abs(x)).mean()  
mse_rw_C_EX = result_C_EX["error"].apply(lambda x: x**2).mean()
rmse_rw_C_EX = np.sqrt(mse_rw_C_EX)  #root mean squared error


# In[151]:


## residuals
result_D_EX["residuals"] = result_D_EX["data"] - result_D_EX["model"]
result_D_EX["error"] = result_D_EX["data"] - result_D_EX["forecast"]
result_D_EX["error_pct"] = result_D_EX["error"] / result_D_EX["data"]
        
## kpi
residuals_mean_D_EX = result_D_EX["residuals"].mean()
residuals_std_D_EX = result_D_EX["residuals"].std()
error_mean_D_EX = result_D_EX["error"].mean()
error_std_D_EX = result_D_EX["error"].std()
mse_D_EX = result_D_EX["error"].apply(lambda x: np.abs(x)).mean()
mape_D_EX = result_D_EX["error_pct"].apply(lambda x: np.abs(x)).mean()  
mse_rw_D_EX = result_D_EX["error"].apply(lambda x: x**2).mean()
rmse_rw_D_EX = np.sqrt(mse_rw_D_EX)  #root mean squared error


# In[152]:


print(f"RMSE Cases (série complète) avec Random walk : {rmse_rw_cases}")
print("")
print(f"RMSE Cases (expérimentale) avec Random walk : {rmse_rw_C_EX}")
print("---------")
print(f"RMSE Deaths avec Random walk : {rmse_rw_deaths}")
print("")
print(f"RMSE Deaths (expérimentale) avec Random walk : {rmse_rw_D_EX}")


#  - ### 8.4 <span style="color:blue"> AutoReg model</span>

#  - #### 8.4.a <span style="color:green"> Choix des lags & Entrainement du modèle</span>

# In[153]:


mod_C_EX = ar_select_order(Cases, maxlag=15)
mod_C_EX.ar_lags


# In[154]:


mod_D_EX = ar_select_order(Deaths, maxlag=15)
mod_D_EX.ar_lags


# In[155]:


mod_C_EX = AutoReg(Cases, lags =[1, 2, 3, 4, 5, 6, 7, 8, 9],seasonal=True)
res_C_EX = mod_C_EX.fit()
print(res_C_EX.summary())


# **Il n y a pas une différence notoire entre l'AIC de la série complète et l'AIC de la série tronquée**.

# In[156]:


mod_D_EX = AutoReg(Deaths, lags =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], seasonal=True)
res_D_EX = mod_D_EX.fit()
print(res_D_EX.summary())


#  - #### 8.4.b <span style="color:green"> AR prediction</span>

# In[157]:


fig = res_C_EX.plot_predict(300,360)
fig = res_D_EX.plot_predict(300,360)


#  - #### 8.4.c <span style="color:green"> Model diagnostics</span>

# In[158]:


fig = res_C_EX.plot_diagnostics(lags=30)


# **La distribution des quantiles est légèrement améliorée par rapport à la série complète et les queues de distribution surtout des nombres de Cas est beaucoup moins épaisse** 

# In[159]:


fig = res_D_EX.plot_diagnostics(lags=30)


#  - ### 8.5 <span style="color:blue"> ARMA model</span>

#  - #### 8.5.a <span style="color:green"> AutoArima</span>

# In[160]:


auto_cases_EX = auto_arima(Cases)
auto_cases_EX.summary()


# **L'AutoArima sur la série nombre de Cas préconise un modèle SARMAX (5,0,4) comme sur la série complète**

# In[161]:


auto_deaths_EX = auto_arima(Deaths)
auto_deaths_EX.summary()


# **L'AutoArima sur la série nombre de Morts préconise un modèle SARMAX (5,0,4) contrairement à la série complète qui était un modèle SARMAX (2,0,2) et avec une saisonnalité de 7 jours visible dans les plots ACF**

#  - #### 8.5.b <span style="color:green"> SARIMAX Fitting</span>

# In[162]:


model_C_EX = sm.tsa.statespace.SARIMAX(train_C_EX, order = (5,0,4), seasonal_order=(1,0,1,7))
model_D_EX = sm.tsa.statespace.SARIMAX(train_D_EX, order = (5,0,4), seasonal_order=(1,0,1,7))


# In[163]:


results_C_EX = model_C_EX.fit()
results_C_EX.summary()


# - **Les lags n'ont pas de significativité statistique avec des p_value>5%**
# - **La saisonnalité 7jours du module AR est la seule qui a une significativté statistique et impacte positivement les valeurs**
# - **L'AIC 5935.044 est amélioré comparé à l'AIC: 9086.248 de la série complète**

# In[164]:


results_D_EX = model_D_EX.fit()
results_D_EX.summary()


# - **Le lag n°4 a une significativité statistique avec une p_value<5% avec une influence négative**
# - **Le MA n°1 a une significativité statistique avec une p_value<5% avec une forte influence négative**
# - **La saisonnalité 7jours du module AR a une significativté statistique et un impact positif important sur les valeurs**
# - **La saisonnalité 7jours du module MA a une significativté statistique et un impact négatif sur les valeurs**
# - **L'AIC 3865.662 est amélioré comparé à l'AIC: 6102.373 de la série complète**

#  - #### 8.5.c <span style="color:green"> SARIMAX Forecast</span>

# In[165]:


pred_df_C_EX = forecast_to_df(results_C_EX, steps = len(test_C_EX))
pred_df_D_EX = forecast_to_df(results_D_EX, steps = len(test_D_EX))


# In[166]:


pred_C_EX = pred_df_C_EX['pred']
pred_D_EX = pred_df_D_EX['pred']


# In[167]:


plot_train_test_pred(train_C_EX,test_C_EX,pred_df_C_EX)


# In[168]:


plot_train_test_pred(train_D_EX,test_D_EX,pred_df_D_EX)


#  - #### 8.5.d <span style="color:green"> RMSE</span>

# In[169]:


mse_C_EX = mean_squared_error(pred_C_EX, test_C_EX)
rmse_C_EX = np.sqrt(mse_C_EX)
mse_D_EX = mean_squared_error(pred_D_EX, test_D_EX)
rmse_D_EX = np.sqrt(mse_D_EX)


#  - ### 8.6 <span style="color:blue"> Bilan des RMSE (série complète & série expérimentale)</span>

# In[170]:


bilan=pd.DataFrame({'Modèles':['Random Walk (Série complète)', 'Random Walk (Série expérimentale)', 'ARMA (Série complète)', 'ARMA (Série expérimentale)','XGBoost'],
                    "Cases":[rmse_rw_cases,rmse_rw_C_EX,rmse_arma_cases,rmse_C_EX,rmsexgb_cases],
                    "Deaths":[rmse_rw_deaths,rmse_rw_D_EX,rmse_arma_deaths,rmse_D_EX,rmsexgb_deaths]})
bilan.set_index("Modèles", inplace=True)


# In[171]:


bilan


# In[ ]:




