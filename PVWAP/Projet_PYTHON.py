#!/usr/bin/env python
# coding: utf-8

# # 1-*INPUT*

# La "source" des données à traiter est sous forme de fichiers .csv comportant la liste des
# trades de bitcoin ayant eu lieu sur différents exchanges de Cryptomonnaies. 
# 
# La premiere partie du programme consistera à importer les **11 fichiers .csv** présents dans le même dossier que le notebook à travers les commandes :
# 
# `glob.glob(chemin des fichiers)` Cette fonction permet de lire tous les fichiers présents sous le chemin donné. 
# 
# 
# `pd.read_csv("nom du fichier.csv")` Cette fonction permettera de définir chaque df à partir des données dans le csv file.
# 
# L'indexation par défaut se faisant de 0 à lignes-1, on utilisera la colonne 'timestamp' déjà incluse dans le fichier comme 
# index avec les arguments de la commande pd.read_csv(`index_col='timestamp', parse_dates=True`)
# 
# 
# Dans la même boucle, et par soucis d'optimisation une partie de la manipulation des données se fait en amont: 
# 
# Les produit des prix & valeurs seront calculés et stockés dans la colonne `[PV]`
# 
# Un resampling selon la fréquence pour calculer la somme horaire des PV et des volumes par la commande:
# `df.resample(frequency).agg({'PV':['sum'],'amount':['sum']}`  avec frequency (5 min, 30 min ou 60 min) comme argument de ré-échantillonage et des fonctions aggrégés qui seront appliqués sur la colonne 'PV' et 'amount'

# In[1]:


#import des librairies
import pandas as pd
import glob


# In[2]:


#Requete d'input pour choisir la fréquence de calcul des métriques
frequency=input("Choisissez une fréquence de calcul: 5 min / 30 min / 60 min")
    
#définition du chemin d'accès
pattern = 'data/*.csv'
csv_files = glob.glob(pattern)

#définition de listes vides
dataframes = []
dfohlc = []
dfstd = []

#boucle qui combine: lecture, traitement et calcul des dataframes
for csv in csv_files:
    #Création d'un à partir fichier csv avec indexation selon la colonne 'timestamp'
    df = pd.read_csv(csv, index_col='timestamp', parse_dates=True)
    
    #Création d'une colonne PV=Price*Amount
    df['PV'] =df['price']*df['amount']
    
    #Création puis stockage d'un autre df avec ré-échantillonage pour avoir l'ohlc 
    df_ohlc=df['price'].resample(frequency).ohlc()
    dfohlc.append(df_ohlc)
    
    #Création puis stockage d'un autre df avec ré-échantillonage  pour avoir l'écart type 
    df_std=df['price'].resample(frequency).std()
    dfstd.append(pd.DataFrame(df_std))
    
    #Ré-échantillonage du df avec calcul des sommes des P*V et la somme des Amounts
    df=df.resample(frequency).agg({'PV':['sum'],'amount':['sum']})
    
    #Créaction d'une colonne Pvwap=Somme des P*V/Somme des Amounts
    df['Pvwap'] =df['PV', 'sum']/df['amount','sum']
    
    #Stockage de chaque df dans la liste des dataframes
    dataframes.append(df)


# # 2-*MANIPULATION*

# In[3]:


dataframes[0].columns


# Maintenant qu'une partie de la manipulation des données est déjà faite, il serait judicieux de 
# 
# tester la présence de valeurs manquantes et choisir quelle méthode appliquer pour les traiter.
# 
# En utilisant la fonction `.isna().sum()` qui affichera la somme des NaN par df

# In[4]:


#Boucle de parcours de la liste dataframes pour test les valeurs manquantes
for x in dataframes:
    print(x.isna().sum())


# Notons la présence de valeurs manquantes dans les dataframes sont dûes à l'absence de transactions à un intervalle de temps donné.
# 
# En assimilant le fait, que le prix n'ayant pas changé par rapport à l'intervalle précédent, on peut considérer la méthode `.fillna(method='ffill')` qui copiera les valeurs précédentes.

# In[5]:


#application de la méthode forwardfill sur le dataframe contenant des NaN
for x in dataframes:
    x=x.fillna(method='ffill')


# # 3-*CALCUL*

# Le Volume weighted average price horaires du bitcoin se calculera avec cette formule:
# 
# ![PWVAP.png](attachment:PWVAP.png)

# Etant donné qu'on a déjà calculé le Pvwap lors de la boucle initiale ainsi que les valeurs OHLC, il ne reste plus qu'à présenter le dataframe final.

# ### Le dataframe final présentera les Pvwap des 11 exchanges et l'index horaire.

# Pour cela on utilisera la fonction `pd.concat` à qui on passera en argument une liste de compréhension `[x[('Pvwap', '')] for x in` 
# 
# `dataframes]` qui prendra que la colonne des Pvwap suivant l'axe des colonnes `axis=1`
# 
# Pour la bonne lecture des données, les Pvwap seront renommés avec le code de leur exchange avec`df.columns `.
# 
# Et finalement, une colonne `[global_Pvwap]` sera ajoutée qui présentera les Pvwap horaires globaux et qui sera égale à, pour 
# 
# une heure donnée, la somme de tous les P*V divisée par la somme de tous les amounts.
# 

# In[6]:


#Concatenation des 11 colonnes Pvwap dans un df 
dff=pd.concat([x[('Pvwap', '')] for x in dataframes],axis=1)

#Renommer chaque colonne par son exchange
dff.columns=['bfly_Pvwap','bfnx_Pvwap','bnus_Pvwap','btrx_Pvwap','cbse_Pvwap','gmni_Pvwap','itbi_Pvwap','krkn_Pvwap','lmax_Pvwap','okcn_Pvwap','stmp_Pvwap']

#Calcul du Pvwap global
for x in dataframes:
    dff['global_Pvwap']=x[('PV', 'sum')].sum()/x[('amount', 'sum')].sum()
    


# # 4-*Dataframe final*

# In[7]:


dff


# #### 4-a Export du dataframe vers un fichier csv

# In[8]:


dff.to_csv(r'export_data\dataframe_final.csv')


# # 5-*Calculs optionnels*

# #### 5-a Standard deviation ou écart type

# In[9]:


#Concatenation des dataframes
dff_std=pd.concat([x for x in dfstd],axis=1)
dff_std.columns=['bfly_std','bfnx_std','bnus_std','btrx_std','cbse_std','gmni_std','itbi_std','krkn_std','lmax_std','okcn_std','stmp_std']

#Test des valeurs manquantes
dff_std.isna().sum()


# L'écart type du premier exchange présente 3 valeurs manquantes, plusieurs méthodes existent pour traiter les valeurs manquantes des écarts types. 
# 
# L'utilisation de la moyenne quotidienne comme substitut des valeurs manquantes sera utilisée pour ne pas fausser la dispersion journalière des écarts types.

# In[10]:


#Remplacement des valeurs manquantes par la moyenne quotidienne
dff_std=dff_std.fillna(dff_std['bfly_std'].mean())


# ##### Dataframe avec écart type

# In[11]:


dff_std


# In[12]:


#Export du dataframe avec écart type

dff_std.to_csv(r'export_data\dff_std.csv')


# #### 5-b OHLC (Open High Low Close values)

# In[13]:


#Concatenation des dataframes stockés dans la liste dfohlc
dff_ohlc=pd.concat([x for x in dfohlc],axis=1)


# In[14]:


#Test des valeurs manquantes
dff_ohlc.isna().sum()


# Les valeurs manquantes étant générées par l'absence de transactions dans l'exchange "bfly" entre 05h et 06h, le prix n'ayant pas changé par rapport à l'heure précédente, on peut reconsidérer la méthode `.fillna(method='ffill')`

# In[15]:


#application de la méthode forwardfill sur le dataframe contenant des NaN
dff_ohlc=dff_ohlc.fillna(method='ffill')
dff_ohlc


# In[16]:


#Export du dataframe avec OHLC
dff_ohlc.to_csv(r'export_data\dff_ohlc.csv')


# #### 5-c Volume weighted median price

# à part les codes issus de la fonction weighted median, il existe des librairies python qui traite la weighted median,
# 
# parmi ces librairies: `Weightedstats`
# 
# qui inclue 4 fonctions (`mean`, `weighted_mean`, `median`, `weighted_median`) qui prennent des listes en arguments
# 
# ainsi que 2 fonctions (`numpy_weighted_mean`, `numpy weighted_median`) qui prennent des listes ou Numpy array en arguments.

# In[ ]:




