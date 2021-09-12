#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
from pandas_datareader import data as pdr
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
import statsmodels.api as sm
from statsmodels.tsa.stattools import kpss
from statsmodels.compat import lzip
from scipy import stats
import statsmodels.stats.diagnostic as sms
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.diagnostic import het_breuschpagan


# In[2]:


#10 actifs choisis parmi les TOP 30 components du DowJones Industrial index
df = yf.download(['VZ', 'GS', 'KO', 'JNJ','INTC','V', 'MCD', 'CAT', 'CVX', 'WMT','^DJI'], start="2010-01-01", end="2021-01-01", interval = '1mo')
df=df['Adj Close'] 
df=df.dropna()
df=df.reset_index()


# In[3]:


#Facteurs
df[['SMB','HML', 'RMW', 'CMA','RF']] = pd.read_excel("data/FamaFrench.xlsx", usecols = "B:F", skiprows = 559, nrows = 132, index_col= None, dtype=np.float64)
df = df.set_index(['Date'])


# In[4]:


df.head()


# ### <span style="color:red"> 1.Transformation des variables </span>

# #### 1.a Transformation des indices boursiers en rendement
# 

# In[5]:


def rendement(x):
    r = ((x - x.shift(1))/(x.shift(1)))*100
    return (r)


# In[6]:


skip = ['RF','SMB','HML', 'RMW', 'CMA'] 
cols = df.columns[~df.columns.isin(skip)]
for i in cols:
    df[i] = rendement(df[i])
df=df.dropna()


# ### <span style="color:red">2.Création des variables dépendantes </span>

# In[7]:


plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['axes.grid'] = False
plt.rcParams['legend.loc'] = 'best'


# #### 2.1 Test de stationnarité du RF

# In[8]:


df['RF'].plot();


# In[9]:


def kpss_test(series, **kw):
    statistic, p_value, n_lags, critical_values= kpss(series, **kw)
    print(f'KPSS statistic: {statistic}')
    print(f'p_value: {p_value}')
    print(f'num lags: {n_lags}')
    print(f'Critical values:')
    for key, value in critical_values.items():
        print(f' {key} : {value}')
    print(f'Result: the series is {"not " if p_value < 0.05 else ""}stationary')


# In[10]:


#Test de stationnarité 
kpss_results = {}
print(f'RF')
kpss_results = kpss_test(df['RF'])


# In[11]:


df['RF'] = df['RF'].diff()
df= df.dropna()


# In[12]:


#Test de stationnarité 
kpss_results = {}
print(f'RF')
kpss_results = kpss_test(df['RF'])


# #### 2.2 Création des variables dépendantes

# #### `y = Le rendement d'un actif (cours boursier en l'occurence) - L'actif sans risque RF`

# In[13]:


pd.options.mode.chained_assignment = None 


# In[14]:


for i in df[cols].columns:
    df[i]=df[i]-df['RF']

y = df[cols].drop("^DJI",1)


# ### <span style="color:red">3.Création de la variable indépendante</span> 
# 
# #### `X= L'indice du DowJones - L'actif sans risque RF`

# In[15]:


df['p_risque']=df['^DJI']-df['RF']


# In[16]:


X = df['p_risque']


# ### <span style="color:red">4.Tests de stationnarité</span> 
# 
# #### 4.1 La variable indépendante

# In[17]:


plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['axes.grid'] = False
plt.rcParams['legend.loc'] = 'best'


# In[18]:


X.plot()
plt.title('Index')
plt.xlabel('Date')
plt.ylabel('P_RISQUE')
plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
plt.show();


# In[19]:


def kpss_test(series, **kw):
    statistic, p_value, n_lags, critical_values= kpss(series, **kw)
    print(f'KPSS statistic: {statistic}')
    print(f'p_value: {p_value}')
    print(f'num lags: {n_lags}')
    print(f'Critical values:')
    for key, value in critical_values.items():
        print(f' {key} : {value}')
    print(f'Result: the series is {"not " if p_value < 0.05 else ""}stationary')


# In[20]:


#Test de stationnarité 
kpss_results = {}
print(f'X')
kpss_results = kpss_test(X)


# #### 4.2 Les variables dépendantes

# In[21]:


y.plot()
plt.title('Index')
plt.xlabel('Date')
plt.ylabel('Rendement')
plt.legend(loc='best', bbox_to_anchor=(1, 0.5), title="Actions")
plt.show();


# In[22]:


#Test de stationnarité
kpss_results = {}
for i in y:
    print(f'{i}')
    kpss_results[i] = kpss_test(y[i])


# ### <span style="color:red">5.Tests de normalité</span> 
# 
# #### 5.1 La variable indépendante X

# In[23]:


X.plot.density();


# In[24]:


X.plot.box();


# [![bxplt-inter.png](https://i.postimg.cc/Jz1YdZcx/bxplt-inter.png)](https://postimg.cc/XpzgrZPB)

# In[25]:


print('QQ plot du p_risque')
sm.qqplot(X, line='s')
plt.show();


# In[26]:


#Test de normalité (Shapiro-Wilk)
#H0 : distribution normale
#HA : pas de distribution normale
p_value = 0.05
a,b= stats.shapiro(X)
print ("Statistiques", a, "p_value", b)
if b < p_value:
    print(f"L'hypothèse nulle H0 peut être rejetée et la série p_risque ne présente pas de distribution normale")
else:
    print(f"L'hypothèse nulle H0 ne peut être rejetée et la série p_risque présente une distribution normale")


#  #### 5.2 Les variables dépendantes y

# In[27]:


y.plot.density();


# In[28]:


y.boxplot();


# In[29]:


for i,val in enumerate(y):
    print('QQ plot de', val)
    sm.qqplot(y[val], line='s')
    plt.show();


# In[30]:


#Test de normalité (Shapiro-Wilk)
#H0 : distribution normale
#HA : pas de distribution normale
p_value = 0.05
for i in y :
    print(i)
    a,b= stats.shapiro(y[[i]])
    print ("Statistiques", a, "p_value", b)
    if b < p_value:
        print(f"L'hypothèse nulle H0 peut être rejetée et la série {i} ne présente pas de distribution normale")
    else:
        print(f"L'hypothèse nulle H0 ne peut être rejetée et la série {i} présente une distribution normale")
        


# ### <span style="color:red">6.Estimation du modèle</span> 

# [![level-level.png](https://i.postimg.cc/V666gLxY/level-level.png)](https://postimg.cc/jWptdKq9)

# In[31]:


Z = pd.concat ([y,X],1)


# In[32]:


skip = ['p_risque'] 
cols = Z.columns[~Z.columns.isin(skip)]
liste = []
alpha = []
pval = []
rsq = []
resid = []
Mco = []
for i in cols:
    mco = smf.ols(i+'~p_risque',data = Z).fit()
    liste.append(mco.summary())
    alpha.append(mco.params.loc['Intercept'])
    pval.append(mco.pvalues)
    rsq.append(mco.rsquared_adj)
    resid.append(sum(mco.resid))
    Mco.append(mco)


# In[33]:


#Résultats de la régression
for l in liste:
    print(f'{l}\n')


# **Analyse des résultats de l'action CAT "CATERPILLAR"**
# 
# - Le **R²** est de 41.6 % qui est un score assez moyen
# - Le p_risque affecte significativement la variable CAT avec une **p-value (P>|t|)** < 0.05 et une **t (coef/std err)** assez élevèe.
# - **Omnibus** décrit la normalité de la distribution des biais combinant les valeurs skew/kurtosis, 0.078 est une distribution pseudo-normale.
# - **Prob(Omnibus)** est la probabilité que les biais auront une distribution normale, 1 correspondrait à une distribution normale, ici 0.962 est une probabilité de distribution pseudo-normale.
# - **Skew** la mesure de symétrie des donnés, 0 serait une symètrie parfaite. **Kurtosis** est la mesure d'aplatissement des données plottées,un Kurtosis important implique un nombre réduit d'outliers.
# - **Durbin-Watson** est la mesure d'homoscédasticité.

# [![level-level2.png](https://i.postimg.cc/sfbYMtWB/level-level2.png)](https://postimg.cc/rzC0bZQc)

# In[34]:


#Test de normalité des résidus(Shapiro-Wilk)
p_value = 0.05
NormofResid = []
for i,MCo in enumerate(Mco):
    a,b= stats.shapiro(MCo.resid)
    if b < p_value:
        NormofResid.append('Les résidus ne suivent pas une loi normale')
    else:
        NormofResid.append('Les résidus suivent une loi normale')
pd.DataFrame(NormofResid, index = cols, columns=['Acceptation du modèle'])


# #### Le modèle est plus pertinent si les valeurs de ses résidus suivent une distribution normale !

# ### <span style="color:red">7. Vérification des hypothèses </span> 

# [![hyp.png](https://i.postimg.cc/sXDBLJhC/hyp.png)](https://postimg.cc/nsgcjB23)

# #### 7.1 Colinéarité

# En Régression linéaire simple, il y a juste une variable, donc la colinéarité n'existe pas.

# #### 7.2 Calcul du Biais

# In[35]:


resid1=pd.DataFrame(resid, index = cols, columns = ['Biais'])
resid1 /= len(y)
resid1


# Des valeurs très proches de zéro !

# #### 7.3 Autocorrelation

# In[36]:


# Test d'autocolération 
name = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
for ii,OlS in enumerate(Mco):
    test=sms.acorr_breusch_godfrey(OlS,20)
    
    if test[1] < p_value:
        print(f"L'hypothèse nulle H0 peut être rejetée et l'actif '{cols[ii]}' est autocorrelé!")
        print('')
    else:
        print(f"L'hypothèse nulle H0 ne peut être rejetée et l'actif '{cols[ii]}' n'est pas autocorrelé!")
        print('')


# #### 7.4 Homoscédasticité

# In[37]:


for i in cols:
    mco = smf.ols(i+'~p_risque',data = Z).fit()
    print(f'== {i} Homoscedasticity Breusch-Pagan Test ==')
    print('')
    print('Breusch-Pagan LM Test Statistic:', np.round(sms.het_breuschpagan(mco.resid, mco.model.exog)[0]))
    print('Breusch-Pagan LM Test P-Value:', np.round(sms.het_breuschpagan(mco.resid, mco.model.exog)[1]))
    print('Breusch-Pagan LM Test F-value:', np.round(sms.het_breuschpagan(mco.resid, mco.model.exog)[2]))
    print('Breusch-Pagan LM Test F P-Value:', np.round(sms.het_breuschpagan(mco.resid, mco.model.exog)[3]))
    p = np.round(sms.het_breuschpagan(mco.resid, mco.model.exog)[1])
    print('----------------')
    if p > 0.05:
        print("Hypothèse H0 retenue (p-value> 5%): Homoscedasticité présente ")
    else:
        print("Hypothèse H0 rejetée (p-value< 5%): pas d'Homoscedasticité présente ")
    print('')


# ### <span style="color:red">8. Vérification de l'Alpha de Jensen </span> 

# In[38]:


CAPM = pd.DataFrame(np.column_stack([rsq, alpha, pval]), index = cols, columns = ('R²_adj','Alpha-Jensen(β0)', 'p-value', 'p_risque'))


# In[39]:


CAPM


# **La p-value** des actifs V (Visa) & MCD (Mc Donald's) montrent une forte significativité statistique du p_risque 

# ### <span style="color:red">9. Application du Fama French 3 facteurs</span> 

# #### 9.1 Mise à jour des variables indépendantes 
# 
# #### `Les 3 facteurs seront "p_risque" / "SMB" / "HML"`

# In[40]:


X_3 = df[['p_risque','SMB','HML']]


# #### 9.2 Tests de stationnarité de X_3 

# In[41]:


X_3.plot()
plt.title('Index')
plt.xlabel('Date')
plt.ylabel('P_RISQUE/SMB/HML')
plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
plt.show();


# In[42]:


#Test de stationnarité 
for i in X_3:
    kpss_results = {}
    print(f'{i}')
    kpss_results = kpss_test(X_3[i])


# #### 9.3 Tests de normalité de X_3 

# In[43]:


X_3.plot.density();


# In[44]:


#Test de normalité (Shapiro-Wilk)
#H0 : distribution normale
#HA : pas de distribution normale
p_value = 0.05
for i in X_3 :
    print(i)
    a,b= stats.shapiro(X_3[[i]])
    print ("Statistiques", a, "p_value", b)
    if b < p_value:
        print(f"L'hypothèse nulle H0 peut être rejetée et la série {i} ne présente pas de distribution normale")
    else:
        print(f"L'hypothèse nulle H0 ne peut être rejetée et la série {i} présente une distribution normale")


# #### 9.4 Vérification des hypothèses des moindres carrées OLS

# In[45]:


Z_3 = pd.concat ([y,X_3],1)


# In[46]:


skip = ['p_risque','SMB','HML'] 
cols = Z_3.columns[~Z_3.columns.isin(skip)]
liste3 = []
alpha3 = []
pval3 = []
rsq3 = []
resid3 = []
Mco = []
for i in cols:
    mco = smf.ols(i+'~p_risque+SMB+HML',data = Z_3).fit()
    liste3.append(mco.summary())
    alpha3.append(mco.params.loc['Intercept'])
    pval3.append(mco.pvalues)
    rsq3.append(mco.rsquared_adj)
    resid3.append(sum(mco.resid))
    Mco.append(mco)


# In[47]:


#Résultats de la régression
for l_3 in liste3:
    print(f'{l_3}\n')


# In[48]:


#Test de normalité (Shapiro-Wilk)
p_value = 0.05
NormofResid = []
for i,MCo in enumerate(Mco):
    a,b= stats.shapiro(MCo.resid)
    if b < p_value:
        NormofResid.append('Les résidus ne suivent pas une loi normale')
    else:
        NormofResid.append('Les résidus suivent une loi normale')
pd.DataFrame(NormofResid, index = cols, columns=['Acceptation du modèle'])


# In[49]:


FF3 = pd.DataFrame(np.column_stack([rsq3, alpha3, pval3]), index = cols, columns = ('R²_adj','Alpha-Jensen(β0)', 'p-value', 'p_risque(β1)', 'SMB(β2)', 'HML(β3)'))


# ### <span style="color:red">10. Vérification des hypothèses </span> 

# #### 10.1 Test de colinéarité

# In[50]:


def vif_cal(input_data, dependent_col):
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=smf.ols(formula="y~x", data=x_vars).fit().rsquared  
        vif=round(1/(1-rsq),2)
        print (xvar_names[i], " VIF = " , vif)


# In[51]:


for col_y in cols:
    print(col_y + ':')
    input = Z_3[[col_y,'p_risque', 'SMB', 'HML']]
    vif_cal(input_data=input , dependent_col= col_y)
    print("-------------------------------")


# #### 10.2 Calcul du Biais

# In[52]:


resid3=pd.DataFrame(resid3, index = cols, columns = ['Biais'])
resid3 /= len(y)
resid3


# #### 10.3 Autocorrelation

# In[53]:


# Test d'autocolération 
name = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
for ii,OlS in enumerate(Mco):
    test=sms.acorr_breusch_godfrey(OlS,20)
    
    if test[1] < p_value:
        print(f"L'hypothèse nulle H0 peut être rejetée et l'actif {cols[ii]} est autocorrelé!")
        print("----------")
    else:
        print(f"L'hypothèse nulle H0 ne peut être rejetée et l'actif {cols[ii]} n'est pas autocorrelé!" )
        print("----------")


# #### 10.4 Homoscédasticité

# In[54]:


for i in cols:
    mco = smf.ols(i+'~p_risque+SMB+HML',data = Z_3).fit()
    print(f'== {i} Homoscedasticity Breusch-Pagan Test ==')
    print('')
    print('Breusch-Pagan LM Test Statistic:', np.round(sms.het_breuschpagan(mco.resid, mco.model.exog)[0]))
    print('Breusch-Pagan LM Test P-Value:', np.round(sms.het_breuschpagan(mco.resid, mco.model.exog)[1]))
    print('Breusch-Pagan LM Test F-value:', np.round(sms.het_breuschpagan(mco.resid, mco.model.exog)[2]))
    print('Breusch-Pagan LM Test F P-Value:', np.round(sms.het_breuschpagan(mco.resid, mco.model.exog)[3]))
    p = np.round(sms.het_breuschpagan(mco.resid, mco.model.exog)[1])
    print('----------------')
    if p > 0.05:
        print("Hypothèse H0 retenue (p-value> 5%): Homoscedasticité présente ")
    else:
        print("Hypothèse H0 rejetée (p-value< 5%): pas d'Homoscedasticité présente ")
    print('')


# ### <span style="color:red">11. Application du Fama French 5 facteurs</span> 

# #### 11.1 Mise à jour des variables indépendantes 
# 
# #### `Les 5 facteurs seront "p_risque" / "SMB" / "HML" / "RMW" / "CMA"`

# In[55]:


X_5 = df[['p_risque','SMB','HML','RMW','CMA']]


# #### 11.2 Tests de stationnarité de X_5 

# In[56]:


# Séparation des plots à différentes échelles

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('RMW / CMA')
ax1.plot(X_5.index, X_5['RMW'], color = 'red', label = 'RMW')
ax1.legend()
ax2.plot(X_5.index, X_5['CMA'], color = 'green',label = 'CMA')
ax2.legend()
plt.show();


# In[57]:


#Test de stationnarité 
for i in X_5[['RMW','CMA']]:
    kpss_results = {}
    print(f'{i}')
    kpss_results = kpss_test(X_5[i])


# #### 11.3 Tests de normalité de X_5 

# In[58]:


X_5['RMW'].plot.density();


# In[59]:


X_5['CMA'].plot.density();


# #### 11.4 Vérification des hypothèses des moindres carrées OLS

# In[60]:


Z_5 = pd.concat ([y,X_5],1)


# In[61]:


skip = ['p_risque','SMB','HML','RMW','CMA'] 
cols = Z_5.columns[~Z_5.columns.isin(skip)]
liste5 = []
alpha5 = []
pval5 = []
rsq5 = []
resid5 = []
Mco = []
for i in cols:
    mco = smf.ols(i+'~p_risque+SMB+HML+RMW+CMA',data = Z_5).fit()
    liste5.append(mco.summary())
    alpha5.append(mco.params.loc['Intercept'])
    pval5.append(mco.pvalues)
    rsq5.append(mco.rsquared_adj)
    resid5.append(sum(mco.resid))
    Mco.append(mco)


# In[62]:


#Test de normalité (Shapiro-Wilk)
p_value = 0.05
NormofResid = []
for i,MCo in enumerate(Mco):
    a,b= stats.shapiro(MCo.resid)
    if b < p_value:
        NormofResid.append('Les résidus ne suivent pas une loi normale')
    else:
        NormofResid.append('Les résidus suivent une loi normale')
pd.DataFrame(NormofResid, index = cols, columns=['Acceptation du modèle'])


# In[63]:


FF5 = pd.DataFrame(np.column_stack([rsq5, alpha5, pval5]), index = cols, columns = ('R²_adj','Alpha-Jensen (β0)', 'p-value', 'p_risque(β1)', 'SMB(β2)', 'HML(β3)', 'RMW(β4)','CMA(β5)'))


# ### <span style="color:red">12. Vérification des hypothèses </span> 

# #### 12.1 Test de colinéarité

# In[64]:


for col_y in cols:
    print(col_y + ':')
    input = Z_5[[col_y,'p_risque','SMB','HML','RMW','CMA']]
    vif_cal(input_data=input , dependent_col= col_y)
    print("--------------")


# #### 12.2 Calcul du Biais

# In[65]:


resid5=pd.DataFrame(resid5, index = cols, columns = ['Biais'])
resid5 /= len(y)
resid5


# #### 12.3 Autocorrelation

# In[66]:


# Test d'autocolération 
name = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
for ii,OlS in enumerate(Mco):
    test=sms.acorr_breusch_godfrey(OlS,20)
    
    if test[1] < p_value:
        print(f"L'hypothèse nulle H0 peut être rejetée et l'actif {cols[ii]} est autocorrelé!")
        print("")
    else:
        print(f"L'hypothèse nulle H0 ne peut être rejetée et l'actif {cols[ii]} n'est pas autocorrelé!")
        print("")


# #### 12.4 Homoscédasticité

# In[67]:


for i in cols:
    mco = smf.ols(i+'~p_risque+SMB+HML+RMW+CMA',data = Z_5).fit()
    print(f'== {i} Homoscedasticity Breusch-Pagan Test ==')
    print('')
    print('Breusch-Pagan LM Test Statistic:', np.round(sms.het_breuschpagan(mco.resid, mco.model.exog)[0]))
    print('Breusch-Pagan LM Test P-Value:', np.round(sms.het_breuschpagan(mco.resid, mco.model.exog)[1]))
    print('Breusch-Pagan LM Test F-value:', np.round(sms.het_breuschpagan(mco.resid, mco.model.exog)[2]))
    print('Breusch-Pagan LM Test F P-Value:', np.round(sms.het_breuschpagan(mco.resid, mco.model.exog)[3]))
    p = np.round(sms.het_breuschpagan(mco.resid, mco.model.exog)[1])
    print('----------------')
    if p > 0.05:
        print("Hypothèse H0 retenue (p-value> 5%): Homoscedasticité présente ")
    else:
        print("Hypothèse H0 rejetée (p-value< 5%): pas d'Homoscedasticité présente ")
    print('')


# ### <span style="color:red">13.  CAPM / Fama & French 3 / Fama & French 5 </span> 

# In[68]:


d = {'CAPM' : CAPM, 'FAMAFRENCH 3' : FF3, 'FAMAFRENCH 5' : FF5}
final = pd.concat([CAPM,FF3,FF5], axis=1, keys=d.keys())


# In[69]:


pd.options.display.max_columns = 20
pd.options.display.max_rows = 20
final


# In[70]:


final.plot(y=[(        'CAPM',            'R²_adj'),('FAMAFRENCH 3',            'R²_adj'),('FAMAFRENCH 5',            'R²_adj')], kind="bar", title='R²_adj');


# - Les **R² ajustés** varient légèrement du CAPM vers le F&F 5 facteurs

# In[71]:


final.plot(y=[(        'CAPM',            'p-value'),('FAMAFRENCH 3',            'p-value'),('FAMAFRENCH 5',            'p-value')], kind="bar",color=['darkred', 'gold', 'mediumblue'],title = 'p-value');


# - Les actifs V (VISA) & MCD (Mc Donald's) ont la plus grande significativité statistique avec une **p-value < 0.05** 

# In[72]:


final.plot(y=[(        'CAPM',            'Alpha-Jensen(β0)'),('FAMAFRENCH 3',            'Alpha-Jensen(β0)'),('FAMAFRENCH 5', 'Alpha-Jensen (β0)')], kind="bar",color=['green', 'black', 'red'], title = 'Alpha-Jensen (β0)');


# - L'**Alpha de Jensen** pour un actif donné augumente du CAPM vers le FF5
