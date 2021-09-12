#!/usr/bin/env python
# coding: utf-8

# ## Import des librairies

# In[2]:


import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import plotly.offline as py
import seaborn as sns
import plotly.graph_objs as go
import plotly 
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# # 1.<span style="color:red"> Lecture des Datasets </span>

# In[2]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# ### 1.1 <span style="color:black">  Concaténation en un DataFrame pour appliquer les mêmes changements</span>
# 

# In[3]:


df = pd.concat([train,test], axis= 0)


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe(include = 'all' )


# # 2.<span style="color:blue"> EDA </span>

# ### 2.1<span style="color:black">  Distribution de la Target </span>

# In[7]:


df['embauche'].value_counts()


# - **On remarque un fort déséquilibre dans la distribution de la classe "embauche" ce qui affectera l'apprentissage si 
#     on ne procède pas à une redistribution de cette variable**

# In[8]:


df['embauche'].value_counts().plot(kind='pie',title= 'distribution de la Target', autopct='%.f%%', legend = False, figsize=(12,6), fontsize=12,explode = [0, 0.2]);


# ### 2.2<span style="color:black"> Pandas profiling du Dataset </span>

# In[ ]:


profile = ProfileReport(df, title="Embauche ou pas")
profile


# **Les NaN & valeurs abérrantes présentes dans ce dataset:**
# 
# - 5 observations dont l'age est supérieur/égal à 70 ans
# - 479 observations dont l'age est inférieur à 16 ans
# - 2 observations dont l'expérience est inférieur à 0
# - 104 observations dont l'expérience est supérieur à l'age
# - 1465 observations dont la note est supérieur à 100.
# - 908 NaN
# 
# <span style="color:blue">**2055 Outliers & 908 NaN soit près de 15% du dataset**</span>

# <span style="color:darkorange"> **Deux méthodologies se présentent:**</span>
#     
#    **1- Supprimer les Outliers & les NaNs**
#     
#    **2- Dans la compétition Kaggle, on était face à une contrainte majeure qui était de garder le set de Test complet à
#        5000 lignes, donc on a procédé à une "harmonisation" des NaN et des valeurs aberrantes**
# 
# 
# 
# 
# <span style="color:blue">**Outliers de la variable "age"**</span>
# - **On procèdera donc à la correction de l'âge en supposant un age minimal légal de travail de 16 ans et maximal de 70 ans**
# 
# 
# <span style="color:blue">**Outliers de la variable "diplome"**</span>
# - **On procèdera donc à l'harmonisation de cette variable en tenant compte de la variable "age" comme suit :**
# 
# **diplome bac--> age 18 ans / license --> 21 ans / master --> 23 ans / doctorat --> 27 ans**
# 
# 
# <span style="color:blue">**Outliers de la variable "note"**</span>
# - **Etant donné le concours d'embauche est noté de 0 à 100, on considérera toutes les notes supérieures à la limite comme arrondie à 100**
# 
# <span style="color:blue">**Outliers de la variable "exp"**</span>
# - **Sur des observations ou l'expérience dépasse l'âge, cette dernière sera remplacée par la moyenne de l'expérience**
# 
# <span style="color:red">**Les valeurs manquantes**</span>
# - **Pour les Nan des variables numériques on imputera la moyenne (mean)**
# - **Pour les Nan des variables catégorielles on imputera le mode (mode)**
# 
# <span style="color:green">**Les variables corrélées**</span>
# -  **Aucune corrélation notoire ou presque n'a été détectée à part Note/Salaire à près de 40%**

# ### 2.3<span style="color:black">  Traitement des outliers </span>

# **Boxplot Diplome/Age**

# In[9]:


plt.figure(figsize=(12,8))
sns.boxplot(x='diplome',
            y='age',
            data=df,
            palette='winter');


# **Boxplot Diplome/Exp**

# In[10]:


plt.figure(figsize=(12,8))
sns.boxplot(x='diplome',
            y='exp',
            data=df,
            palette='winter');


# **Boxplot Exp/Age**

# In[11]:


plt.figure(figsize=(12,8))
sns.boxplot(x='exp',
            y='age',
            data=df,
            palette='winter');


# In[12]:


#------------#
df.loc[(df['age'] >= 70), 'age'] = round(df['age'].mean(), 0) #5 Observations
df.loc[(df['age'] < 16), 'age'] = round(df['age'].mean(), 0) #479 Observations
#------------#
df.loc[(df['diplome'] == "bac"), 'age'] = 18 #2453 observations
df.loc[(df['diplome'] == "licence"), 'age'] = 21 #7377 observations
df.loc[(df['diplome'] == "master"), 'age'] = 23 #7513 observations
df.loc[(df['diplome'] == "doctorat"), 'age'] = 27 #2547 observations
#------------#
df.loc[(df['exp'] < 0), 'exp'] = round(df['exp'].mean(), 0) #2 observations
df.loc[(df['exp'] > df['age']),'exp'] = round(df['exp'].mean(),0) #104 observations
#------------#
df.loc[(df['note'] > 100), 'note'] = 100 #1465 observations
#------------#


# ### 2.4<span style="color:black">  Traitement des NAN </span>

# In[13]:


plt.figure(figsize=(12,8))
sns.heatmap(df.isnull(), 
            yticklabels=False, 
            cbar=False, 
            cmap='viridis');


# In[14]:


#------Variables Numériques-------#
NUMERICAL = ["age","exp","salaire","note"]
df[NUMERICAL]= df[NUMERICAL].astype(np.float32)
df[NUMERICAL] = df[NUMERICAL].fillna(round(df[NUMERICAL].mean(), 0))

#------Variables Catégorielles-------#
CATEGORICAL = ["cheveux","sexe","diplome","specialite","dispo","date"]
df[CATEGORICAL]= df[CATEGORICAL].astype('category')
df[CATEGORICAL] = df[CATEGORICAL].fillna(df[CATEGORICAL].mode().iloc[0])


# ### 2.5<span style="color:black">  Création de nouvelles features numériques à partir de la date </span>

# In[15]:


df['date'] = pd.to_datetime(df['date'],format="%Y-%m-%d")
df['year']=  df['date'].dt.year
df['month']=  df['date'].dt.month
df['day']=  df['date'].dt.day


# ### 2.6 <span style="color:black">  Création de nouvelles features catégoriques </span>

# In[16]:


df['q_exp'] = pd.qcut(df['exp'],q=3,precision=0)
df['q_age'] = pd.qcut(df['age'], q=3,precision=0)
df['q_note'] = pd.qcut(df['note'],q=4,precision=0)
df['q_salaire'] = pd.qcut(df['salaire'],q=5,precision=0)


# ### 2.4 <span style="color:black"> Redéfinition des Variables numériques/catégorielles/features/Target </span>

# In[17]:


NUMERICAL = ["age","exp","salaire","note","year","month","day"]
df[NUMERICAL]= df[NUMERICAL].astype(np.float32)


# In[18]:


CATEGORICAL = ["cheveux","sexe","diplome","specialite","dispo"]
df[CATEGORICAL]= df[CATEGORICAL].astype('category')


# In[19]:


FEATURES = NUMERICAL + CATEGORICAL + ["q_exp","q_age","q_note",'q_salaire']
TARGET = "embauche"


# ### 2.5 <span style="color:black"> Data Viz </span>

# **Distribution des classes de la variable AGE par rapport à la TARGET**

# In[20]:


plt.figure(figsize=(14,6))
plt.hist(df[df["embauche"]==1]["age"], edgecolor="k",density=True, alpha=0.7, label = "Embauché(e)")
plt.hist(df[df["embauche"]==0]["age"], edgecolor="k",density=True, alpha=0.7, label = "Pas embauché(e)")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# **Distribution des classes de la variable EXP par rapport à la TARGET**

# In[21]:


plt.figure(figsize=(14,6))
plt.hist(df[df["embauche"]==1]["exp"], edgecolor="k",density=True, alpha=0.7, label = "Embauché(e)")
plt.hist(df[df["embauche"]==0]["exp"], edgecolor="k",density=True, alpha=0.7, label = "Pas embauché(e)")
plt.xlabel("Experience")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# **Distribution des classes de la variable NOTE par rapport à la TARGET**

# In[22]:


plt.figure(figsize=(14,6))
plt.hist(df[df["embauche"]==1]["note"], edgecolor="k",density=True, alpha=0.7, label = "Embauché(e)")
plt.hist(df[df["embauche"]==0]["note"], edgecolor="k",density=True, alpha=0.7, label = "Pas embauché(e)")
plt.xlabel("Note")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# **Distribution des classes de la variable SALAIRE par rapport à la TARGET**

# In[23]:


plt.figure(figsize=(14,6))
plt.hist(df[df["embauche"]==1]["salaire"], edgecolor="k",density=True, alpha=0.7, label = "Embauché(e)")
plt.hist(df[df["embauche"]==0]["salaire"], edgecolor="k",density=True, alpha=0.7, label = "Pas embauché(e)")
plt.xlabel("Salaire")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# **Distribution des classes de la variable YEAR par rapport à la TARGET**

# In[24]:


plt.figure(figsize=(14,6))
sns.countplot(data=df, x="year",hue="embauche", edgecolor="k")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()


# **Distribution des classes de la variable MONTH par rapport à la TARGET**

# In[25]:


plt.figure(figsize=(14,6))
sns.countplot(data=df, x="month",hue="embauche", edgecolor="k")
plt.xlabel("Month")
plt.ylabel("Count")
plt.show()


# **Distribution des classes de la variable DAY par rapport à la TARGET**

# In[26]:


plt.figure(figsize=(14,6))
sns.countplot(data=df, x="day",hue="embauche", edgecolor="k")
plt.xlabel("day")
plt.ylabel("Count")
plt.show()


# **Distribution de la variable CHEVEUX par rapport à la TARGET**

# In[27]:


plt.figure(figsize=(14,6))
sns.countplot(data=df, x="cheveux",hue="embauche", edgecolor="k")
plt.xlabel("Cheveux")
plt.ylabel("Count")
plt.show()


# **Distribution de la variable DIPLOME par rapport à la TARGET**

# In[28]:


plt.figure(figsize=(14,6))
sns.countplot(data=df, x="diplome",hue="embauche", edgecolor="k")
plt.xlabel("Diplome")
plt.ylabel("Count")
plt.show()


# **Distribution de la variable SPECIALITE par rapport à la TARGET**

# In[29]:


plt.figure(figsize=(14,6))
sns.countplot(data=df, x="specialite",hue="embauche", edgecolor="k")
plt.xlabel("specialite")
plt.ylabel("Count")
plt.show()


# **Distribution de la variable DISPO par rapport à la variable SEXE**

# In[30]:


plt.figure(figsize=(14,6))
sns.countplot(data=df, x="dispo",hue="embauche", edgecolor="k")
plt.xlabel("Dispo")
plt.ylabel("Count")
plt.show()


# ### 2.6 <span style="color:black"> Tests Statistiques </span>

# In[31]:


import scipy


# **CHEVEUX / SALAIRE**
# - Hypothèse H0 : Pas de relation statistiquement significative

# In[32]:


data_blond =df[df["cheveux"]=="blond"]
data_brun = df[df["cheveux"]=="brun"]
data_roux =df[df["cheveux"]=="roux"]
data_chatain =df[df["cheveux"]=="chatain"]
stat, p_value = scipy.stats.kruskal(data_blond["salaire"], data_brun["salaire"],data_roux["salaire"] ,data_chatain["salaire"])

print('Statistics=%.3f, p_value=%.3f' % (stat, p_value))
# interpret
alpha = 0.05
if p_value > alpha:
    print('Même distributions (Hypothèse H0 non rejetée)')
else:
    print('Distributions différentes (Hypothèse H0 rejetée)')


# **SPECIALITE / SEXE**
# - Hypothèse H0 : Pas de relation statistiquement significative

# In[33]:


data_forage =df[df["specialite"]=="forage"]
data_geologie = df[df["specialite"]=="geologie"]
data_detective =df[df["specialite"]=="detective"]
data_archeologie =df[df["specialite"]=="archeologie"]
stat, p_value = scipy.stats.kruskal(data_forage["sexe"], data_geologie["sexe"],data_detective["sexe"] ,
                                    data_archeologie["sexe"])

print('Statistics=%.3f, p_value=%.3f' % (stat, p_value))
# interpret
alpha = 0.05
if p_value > alpha:
    print('Même distributions (Hypothèse H0 non rejetée)')
else:
    print('Distributions différentes (Hypothèse H0 rejetée)')


# **EXP / NOTE**
# - Hypothèse H0 : Pas de relation statistiquement significative

# In[34]:


data_exp =df["exp"]
data_note = df["note"]
stat, p_value = scipy.stats.kruskal(data_exp, data_note)

print('Statistics=%.3f, p_value=%.3f' % (stat, p_value))
# interpret
alpha = 0.05
if p_value > alpha:
    print('Même distributions (Hypothèse H0 non rejetée)')
else:
    print('Distributions différentes (Hypothèse H0 rejetée)')


# In[35]:


plt.figure(dpi=150)
sns.heatmap(df.corr('spearman'),annot=False,cmap='rocket',lw=1);


# In[36]:


from scipy.stats import chi2_contingency


# In[37]:


def test_chi_2(QualVar,target,alpha):

   

    QualVar = pd.DataFrame(QualVar)

    liste_chi2 = []

    liste_chi2_name = []

   

    # ici on créé le tableau de contingence pour réaliser notre test :

        

    for i in range(len(list(QualVar.columns))):

 

        table = pd.crosstab(QualVar[list(QualVar.columns)[i]],QualVar[target])

        stat, p, dof, expected = chi2_contingency(table)

 
        if p <= alpha:

            liste_chi2.append(i)

        else:

            pass
   

    for j in liste_chi2:

        liste_chi2_name.append([i.encode('ascii', 'ignore') for i in QualVar.columns][j])

       

    return liste_chi2_name


# In[38]:


liste_chi2_name = test_chi_2(df,"embauche",0.05)
liste_chi2_name


# Les variables listées ci-dessus ont une p_value< 5% et donc présente une significativité statistique pour expliquer la TARGET

# # 3.<span style="color:green"> PREPROCESSING </span>

# ### 3.1<span style="color:black">  Label Encoding </span>

# **Le choix s'est porté sur le label encoding pour éviter une augumentation de la dimension créée par le One hot encoding par exemple, et ce pour plus de performance lors des Tunnings des hyperparamètres**

# In[39]:


df_c=df.copy()


# In[40]:


label_encoder = preprocessing.LabelEncoder()
df_c[CATEGORICAL]=df[CATEGORICAL].apply(label_encoder.fit_transform)
df_c[["q_exp","q_age","q_note",'q_salaire']] = df[["q_exp","q_age","q_note",'q_salaire']].apply(label_encoder.fit_transform)
df_c[TARGET]=df[TARGET]


# ### 3.2<span style="color:black"> Transformation du type </span>

# In[41]:


df_c['age'] = df_c['age'].astype(np.uint8)
df_c['exp'] = df_c['exp'].astype(np.uint8)
df_c['salaire'] = df_c['salaire'].astype(np.uint8)
df_c['cheveux'] = df_c['cheveux'].astype(np.uint8)
df_c['note'] = df_c['note'].astype(np.float16)
df_c['sexe'] = df_c['sexe'].astype(np.uint8)
df_c['diplome'] = df_c['diplome'].astype(np.uint8)
df_c['specialite'] = df_c['specialite'].astype(np.uint8)
df_c['dispo'] = df_c['dispo'].astype(np.uint8)
df_c['year'] = df_c['year'].astype(np.int16)
df_c['month'] = df_c['month'].astype(np.int16)
df_c['day'] = df_c['day'].astype(np.int16)
df_c['q_exp'] = df_c['q_exp'].astype(np.int16)
df_c['q_age'] = df_c['q_age'].astype(np.int16)
df_c['q_salaire'] = df_c['q_salaire'].astype(np.int16)    
df_c['q_note'] = df_c['q_note'].astype(np.int16) 


# ### 3.3<span style="color:black"> Train/Test Split </span>

# In[42]:


train = df_c.loc[~df_c[TARGET].isna()]


# In[43]:


test = df_c.loc[df_c[TARGET].isna()]


# ### 3.4<span style="color:black"> Oversampling de la classe minoritaire "embauche = 1" </span>

# **Le SMOTETomek procédera à la création de valeurs synthétiques similaires aux vraies valeurs présentes dans le dataset avec une Embauche = 1**

# In[44]:


from imblearn.combine import SMOTETomek


# In[45]:


smotetomek_X = train[FEATURES]
smotetomek_Y = train[TARGET]

smote_tomek = SMOTETomek(random_state=68, sampling_strategy=0.99) #La classe 1 sera 99% de la classe 0
X_resampled, y_resampled = smote_tomek.fit_resample(train[FEATURES], train[TARGET])

smotetomek_X = pd.DataFrame(data = X_resampled,columns=FEATURES)
smotetomek_Y = pd.DataFrame(data = y_resampled,columns=['embauche'])
print ((smotetomek_Y['embauche'] == 1).sum())
print ((smotetomek_Y['embauche'] == 0).sum())


# In[46]:


train_X = smotetomek_X.copy()


# In[47]:


train_Y = smotetomek_Y.copy()


# In[48]:


train_X = train_X[FEATURES]
train_Y = train_Y[TARGET]
test_X = test[FEATURES]


# In[49]:


df_oversampler = pd.concat([train_X,train_Y], axis= 1)


# **Distribution de la target après Oversampling**

# In[50]:


df_oversampler['embauche'].value_counts().plot(kind='pie',title= 'distribution de la Target', autopct='%.f%%', legend = False, figsize=(12,6), fontsize=12,explode = [0, 0.2]);


# ### 3.4<span style="color:black"> Standardisation des données</span>

# **Remarque** : 
# 
# **La standardisation des données n'est pas nécessaire quand on utilise des algorithmes d'apprentissage non sensibles à l'amplitude des variables tels que**
# - La régression logistique
# - Le Random Forest
# - Les modèles de Gradient boosting
# 
# **Hors dans ce projet, on utilisera aussi le SVC, DTC & KNN qui eux sont sensibles à l'amplitude des variables**

# In[51]:


train_X.std()


# In[52]:


test_X.std()


# In[53]:


scaler = StandardScaler()

train_X = scaler.fit_transform(train_X)
test_X = scaler.fit_transform(test_X)


# In[54]:


train_X = train_X.astype('float32')
test_X = test_X.astype('float32')


# # 4.<span style="color:Orange"> MODELISATION </span>

# - Le projet présenté à pour but une classification de la TARGET entre 0 & 1
# 
# - On choisira donc des Algorithmes d'apprentissage supervisé pour CLASSIFICATION
# 
# - Régression Logistique /Decision Tree/ SVC / KNN / Random Forest / Gradient boosting / XGBoost
# 
# - La comparaison des modèles se fera principalement sur le score AUC
# 
# - Le tunning des hyperparamètres se fera avec HalvingGridSearchCV qui est une nouvelle classe de tunning des hyperparamètres beaucoup plus rapide que le GridsearchCV avec pratiquement les mêmes résultats

# ### 4.1<span style="color:black"> Tunning des Hyperparamètres avec HalvingGridSearchCV </span>

# In[55]:


def tunning(param_grid,model,X,Y):    
    halving = HalvingGridSearchCV(model, param_grid = param_grid,scoring="roc_auc", min_resources = "exhaust",
                                  n_jobs = -1,cv = 5, factor = 3, verbose = 1)
    halving.fit(X, Y)
    print ("Best Score: {}".format(halving.best_score_)) 
    print ("Best params: {}".format(halving.best_params_)) 


# ### 4.2<span style="color:black"> Evaluation du modèle </span>

# In[56]:


def evaluation(model,z,X,Y):
    model.fit(X,Y)
    predict   = model.predict(X)
    proba = model.predict_proba(X)
    fig = plt.figure()
    #roc_auc_score
    model_roc_auc = metrics.roc_auc_score(Y,predict) 
    #Confusion matrix
    conf_matrix = metrics.confusion_matrix(Y,predict)
    #plot confusion matrix
    plot1 = go.Heatmap(z = conf_matrix ,
                        x = ["Pred_0","Pred_1"],
                        y = ["Real_0","Real_1"],
                        showscale  = True,autocolorscale = True,
                        name = "matrix", transpose = True, visible =  True)
    #plot roc auc
    a,b,c = metrics.roc_curve(Y,proba[:,1])
    plot2 = go.Scatter(x = a,y = b,
                        name = "Roc : " + str(model_roc_auc),
                        line = dict(color = ('rgb(22, 96, 167)'),width = 2))
    plot3 = go.Scatter(x = [0,1],y=[0,1],
                        line = dict(color = ('rgb(205, 12, 24)'),width = 2,
                        dash = 'dot'))
    #plot coefficients/Features
    if z == "coefficients" :
        coefficients  = pd.DataFrame(model.coef_.ravel())
    elif z== "features" :
        coefficients  = pd.DataFrame(model.feature_importances_)
    column_df     = pd.DataFrame(FEATURES)
    coef_sumry    = (pd.merge(coefficients,column_df,left_index= True,
                              right_index= True, how = "left"))
    coef_sumry.columns = ["coefficients","features"]
    coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)
    plot4 = trace4 = go.Bar(x = coef_sumry["features"],y = coef_sumry["coefficients"],
                    name = "coefficients",
                    marker = dict(color = coef_sumry["coefficients"],
                                  colorscale = "Picnic",
                                  line = dict(width = .6,color = "black")))


    #Subplots

    fig = plotly.subplots.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                            subplot_titles=('Confusion Matrix',
                                            'Receiver operating characteristic',
                                            'Feature Importances'),print_grid=False)
    fig.append_trace(plot1,1,1)
    fig.append_trace(plot2,1,2)
    fig.append_trace(plot3,1,2)
    fig.append_trace(plot4,2,1)
    fig['layout'].update(showlegend=False, title="Model performance" ,
                         autosize = False,height = 900,width = 800,
                         plot_bgcolor = 'rgba(240,240,240, 0.95)',
                         paper_bgcolor = 'rgba(240,240,240, 0.95)',
                         margin = dict(b = 195))
    fig["layout"]["xaxis2"].update(dict(title = "false positive rate"))
    fig["layout"]["yaxis2"].update(dict(title = "true positive rate"))
    fig["layout"]["xaxis3"].update(dict(showgrid = True,tickfont = dict(size = 10),
                                        tickangle = 90))
    py.iplot(fig);
    print ("ROC-AUC : ",model_roc_auc,"\n")
    print("score F1 : ", metrics.f1_score(Y, predict),"\n")
    print ("Accuracy Score : ",metrics.accuracy_score(Y,predict))   


# In[57]:


def evaluation_knn(model,X,Y):
    model.fit(X,Y)
    predict   = model.predict(X)
    proba = model.predict_proba(X)
    #roc_auc_score
    model_roc_auc = metrics.roc_auc_score(Y,predict) 
    #plot confusion matrix
    plot_confusion_matrix(model, X, Y)  
    plt.show();
    print ("ROC-AUC : ",model_roc_auc,"\n")
    print("score F1 : ", metrics.f1_score(Y, predict),"\n")
    print ("Accuracy Score : ",metrics.accuracy_score(Y,predict))


# In[58]:


def MetricsMaker(model):
    # Save Models
    # Splits
    kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=2021)
    split = list(kf.split(train_X,train_Y))
    Metrics = {}
    Precision, Accuracy, F1_score, Recall_score, ROC_AUC = 0, 0, 0, 0, 0
    for i,(train_index, test_index) in enumerate(split):

        data_train = train_X[train_index] 
        y_train = train_Y[train_index] 
        data_test = train_X[test_index]
        y_test = train_Y[test_index]

        # create a fitted model
        fittedModel = model.fit(data_train,y_train)
        y_hat_proba = fittedModel.predict_proba(data_test)[:,1]
        y_hat = fittedModel.predict(data_test)
        #  log_l = 
        Precision += metrics.precision_score(y_test,y_hat)
        Accuracy += metrics.accuracy_score(y_test,y_hat)
        F1_score += metrics.f1_score(y_test,y_hat)
        Recall_score += metrics.recall_score(y_test,y_hat)
        ROC_AUC += metrics.roc_auc_score(y_test,y_hat)
        
    Metrics['Precision'] = Precision / 5 
    Metrics['Accuracy'] = Accuracy / 5
    Metrics['F1_score'] = F1_score / 5
    Metrics['Recall_score'] = Recall_score / 5
    Metrics['ROC-AUC'] = ROC_AUC / 5
    
    return Metrics


# In[59]:


# Les metrics scores de chaque modeles seront stockés ici!
Metrics = {}


# ### 4.2<span style="color:black"> Régression Logistique </span>

# In[60]:


parameters = {'Cs': [1, 2, 3, 4, 5, 6 ,7 ,8 ,9 ,10]
             }

logit = LogisticRegressionCV(random_state= 33,cv=10,max_iter=10000,verbose=1, n_jobs = -1)

#tunning(parameters,logit,train_X,train_Y)


# In[61]:


logReg = LogisticRegressionCV(Cs= 6, random_state= 33,cv=10,max_iter=10000,verbose=1)
Metrics['LogisticRegressionCV'] = MetricsMaker(logReg)


# In[62]:


#Evaluation avec le modèle tunné
logit = LogisticRegressionCV(Cs= 6, random_state= 33,cv=10,max_iter=10000,verbose=1)
evaluation(logit,"coefficients",train_X,train_Y)  


# ### 4.3<span style="color:black"> Decision Tree Classifier </span>

# In[63]:


d_t_c = DecisionTreeClassifier(random_state=33)
parameters = {'max_depth': [1, 2, 3, 4, 5, 6, 7],
              'max_features': [1, 2, 3, 4, 5],
              'criterion': ['gini','entropy'],
              'splitter': ['best'],
              }
    
#tunning(parameters,d_t_c,train_X,train_Y.values.ravel())


# In[64]:


D_T_C =  DecisionTreeClassifier(random_state=33, criterion = "gini", max_depth=7, max_features = 5, splitter = "best")
Metrics['DecisionTreeClassifier'] = MetricsMaker(D_T_C)


# In[65]:


#Evaluation avec le modèle tunné
d_t_c =  DecisionTreeClassifier(random_state=33, criterion = "gini", max_depth=7, max_features = 5, splitter = "best")

evaluation(d_t_c,"features",train_X,train_Y)


# ### 4.4<span style="color:black"> SVC </span>

# **Le Tunning s'est fait un hyperparamètre à la fois malgrè que cela peut fausser les meilleurs combinaisons mais pour éviter une attente trop longue lors de l'execution**

# In[66]:


s_v_c  = SVC(random_state=33,verbose=2)
parameters = {'kernel': ["linear","rbf","poly"], 
              'gamma': [0.1, 1, 10, 100],
              'C': [0.1, 1, 10, 100,1000],
              'degree': [0, 1, 2, 3, 4, 5, 6]
              }
#tunning(parameters,s_v_c,train_X,train_Y.values.ravel())


# In[67]:


S_V_C =  SVC(random_state=33, kernel = "rbf", gamma=0.1, C = 10, degree = 4,probability=True,verbose=2 ) 
Metrics['SVC'] = MetricsMaker(S_V_C)


# In[68]:


#Evaluation avec le modèle tunné
s_v_c =  SVC(random_state=33, kernel = "rbf", gamma=0.1, C = 10, degree = 4,probability=True,verbose=2 ) 

evaluation_knn(s_v_c,train_X,train_Y) #Since rbf Kernel is used


# ### 4.5<span style="color:black"> KNN Classifier </span>

# In[69]:


k_n_n = KNeighborsClassifier(algorithm='auto', n_jobs = -1)

parameters = {
    'leaf_size':[5,10,20,30], 
    'n_neighbors':[3,4,5,8,10,11,12],
    'weights' : ['uniform', 'distance'],
    'p' : [1,2]
}

#tunning(parameters,k_n_n,train_X,train_Y)


# In[70]:


K_N_N = KNeighborsClassifier(algorithm='auto',leaf_size= 20,n_neighbors= 11, p=1, weights = "distance", n_jobs = -1)
Metrics['KNeighborsClassifier'] = MetricsMaker(K_N_N)


# In[71]:


#Evaluation avec le modèle tunné
k_n_n = KNeighborsClassifier(algorithm='auto',leaf_size= 20,n_neighbors= 11, p=1, weights = "distance", n_jobs = -1)

evaluation_knn(k_n_n,train_X,train_Y)


# ### 4.6<span style="color:black"> Random Forest Classifier </span>

# In[72]:


r_f_c = RandomForestClassifier(random_state=33, verbose=2,n_jobs = -1)
parameters = {
    'n_estimators': [5,10,15,20,30,40,50,60,70,80],
    'min_samples_split': [3, 5, 10], 
    'max_depth': [2, 5, 15, 30,50,70,80],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False],
    'criterion': ['gini','entropy']   
}


#tunning(parameters,r_f_c,train_X,train_Y.values.ravel())


# In[73]:


R_F_C = RandomForestClassifier(random_state=33, verbose=2, n_estimators = 70,
                               min_samples_split= 3, max_depth = 70, max_features = "auto",
                              bootstrap = "False", criterion = "gini")
Metrics['RandomForestClassifier'] = MetricsMaker(R_F_C)


# In[74]:


#Evaluation avec le modèle tunné
r_f_c = RandomForestClassifier(random_state=33, verbose=2, n_estimators = 70,
                               min_samples_split= 3, max_depth = 70, max_features = "auto",
                              bootstrap = "False", criterion = "gini")

evaluation(r_f_c,"features",train_X,train_Y)


# ### 4.7<span style="color:black"> Gradient boosting Classifier </span>

# In[75]:


g_b_c = GradientBoostingClassifier (random_state = 33, verbose=2)
parameters = {'learning_rate'    : [0.01,0.02,0.03,0.04,0.06,0.08,0.09],
                  'loss'         : ["deviance", "exponential"],
                  'subsample'    : [0.9, 0.5, 0.2, 0.1],
                  'n_estimators' : [100,500,1000, 1500],
                  'max_depth'    : [4,6,8,10],
                  'criterion'    : ["friedman_mse", "mse"],
                  'min_samples_split' : [2,4,6,8,10,12,14],
                  'min_samples_leaf'  : [1,2,3,4],
                  'max_features'      : ["auto", "sqrt", "log2"]
            }

#tunning(parameters,g_b_c,train_X,train_Y.values.ravel())


# In[76]:


G_B_C = GradientBoostingClassifier(learning_rate=0.09, n_estimators=500, max_depth = 8, min_samples_split = 12, 
         max_features='auto', subsample=0.1,criterion= "friedman_mse", min_samples_leaf = 2,
         loss = "exponential", random_state=33, verbose = 1)
Metrics['GradientBoostingClassifier'] = MetricsMaker(G_B_C)


# In[77]:


#Evaluation avec le modèle tunné
g_b_c = GradientBoostingClassifier(learning_rate=0.09, n_estimators=500, max_depth = 8, min_samples_split = 12, 
         max_features='auto', subsample=0.1,criterion= "friedman_mse", min_samples_leaf = 2,
         loss = "exponential", random_state=33, verbose = 1)
evaluation(g_b_c,"features",train_X,train_Y)


# ### 4.8<span style="color:black"> XGBoost Classifier </span>

# In[78]:


x_g_c = XGBClassifier(use_label_encoder=False)

parameters = {'nthread':[4,5,6,8,10,12], 
              'learning_rate': [0.01,0.03,0.05,0.1,0.2,0.3,0.4,0.5],
              'max_depth': range (2, 21, 1),
              'min_child_weight': [10,12,14,16,18,20],
              'subsample': [0.6,0.8,1],
              'colsample_bytree': [0.2,0.4,0.5,0.7],
              'n_estimators': [100,200,300,400,500] 
              }

#tunning(parameters,x_g_c,train_X,train_Y.values.ravel())


# In[79]:


X_G_B = XGBClassifier(learning_rate = 0.4,nthread = 10,max_depth = 16, subsample=0.8,colsample_bytree=0.5
                      ,n_estimators = 200, min_child_weight = 16,
              use_label_encoder=False, random_state = 33, verbosity=1)
Metrics['XGBClassifier'] = MetricsMaker(X_G_B)


# In[80]:


#Evaluation avec le modèle tunné
x_g_c = XGBClassifier(learning_rate = 0.4,nthread = 10,max_depth = 16, subsample=0.8,colsample_bytree=0.5
                      ,n_estimators = 200, min_child_weight = 16,
              use_label_encoder=False, random_state = 33, verbosity=1)

evaluation(x_g_c,"features",train_X,train_Y.values.ravel())


# # 5.<span style="color:Turquoise"> FEATURES SELECTION </span>

# ### 5.1<span style="color:black"> Select KBest  </span>

# In[81]:


kbest = SelectKBest(score_func=f_classif, k='all') #Score_func peut etre f_classif ou chi2
fit = kbest.fit(train_X, train_Y.values.ravel())


# In[82]:


np.set_printoptions(precision=3) #Chaque score correspond à une colonne, les variables a retenir sont celles qui ont le meilleur score
d = { label: value for label, value in zip(FEATURES, fit.scores_) }
d


# ### 5.1<span style="color:black"> RFECV avec XGboost Classifier tunné  </span>

# In[83]:


train_X = pd.DataFrame(train_X, columns = FEATURES)


# In[84]:


rfecv = RFECV(estimator=x_g_c,cv=5,scoring="f1")   ## on peut choisir le min_features_to_select( 1 par défaut)
rfecv = rfecv.fit(train_X, train_Y.values.ravel())

print('Nombre optimal de variables :', rfecv.n_features_)
print('Les meilleures variables :', train_X.columns[rfecv.support_])
best_features = list(train_X.columns[rfecv.support_])


# # 5.<span style="color:Purple"> PREDICTION </span>

# **Les prédictions de la base test se feront avec chaque modèle tunné pour pouvoir comparer le meilleur modèle de classification**

# **Les métriques de comparaison**
# 
# `recall` : Nombre de classes trouvées par rapport aux nombres entiers de cette même classe.
# 
# `precision` : Combien de classes ont été correctements classifiées
# 
# `f1-score` : La moyenne harmonique entre precision & recall

# ## Comparaison

# In[85]:


pd.DataFrame(Metrics)

