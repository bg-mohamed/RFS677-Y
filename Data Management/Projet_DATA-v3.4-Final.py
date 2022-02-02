#!/usr/bin/env python
# coding: utf-8

# In[6]:


#import des librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import seaborn as sns
import math 
import warnings
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')



# ## INPUT

# In[2]:


#Lecture du fichier CSV en notifiant le delimiteur comme étant point-virgule
df = pd.read_csv(r'du-vert-pres-de-chez-moi.csv', sep=';')


# In[3]:


# L'utilisation de la mémoire est réduite
df['Arrondissement'] = pd.to_numeric(df['Arrondissement'], downcast='signed')


# In[4]:


df.head()


# In[5]:


print(f"La base fait {df.shape[0]} lignes et {df.shape[1]} colonnes")


# ## Preprocessing

# In[6]:


#Suppression des zéros dans la colonne arrondissement

df.drop(df.loc[df['Arrondissement']==0.0].index, inplace=True)


# In[7]:


#Valeurs uniques

Numero = pd.DataFrame(df['Numéro'].value_counts())

Numero


# In[8]:


PotentialDoublons = Numero[Numero['Numéro'] > 1]
PotentialDoublons


# In[9]:


def samePlace(num1, num2,code):
    dx = float(df[df['Numéro'] == code]['Geo Point'].iloc[num1].split(',')[0]) - float(df[df['Numéro'] == code]['Geo Point'].iloc[num2].split(',')[0])
    dy = float(df[df['Numéro'] == code]['Geo Point'].iloc[num1].split(',')[1]) - float(df[df['Numéro'] == code]['Geo Point'].iloc[num2].split(',')[1])
    if math.sqrt(dx**2 + dy**2) < 10**-3 : return True
    return False


# In[10]:


# Les lignes qui ont le même numéro et leur coordonnés sont identiques, ils sont considérés comme des doublons!
for _,code in enumerate(PotentialDoublons.index):
    num = PotentialDoublons['Numéro'].loc[code]
    while(num > 1):
        if samePlace(num - 1,num - 2,code):
            if(df[df['Numéro'] == code]['Avancement du projet'].iloc[num - 2] == '-'): 
                df.drop(df[df['Numéro'] == code].index[num - 2], inplace=True)
            else : 
                df.drop(df[df['Numéro'] == code].index[num - 2], inplace=True)
        num -= 1


# In[11]:


df.sort_values(by = 'Arrondissement', inplace = True)


# In[12]:


df.drop(['Geo Shape', 'Adresse', 'Numéro'],axis = 1, inplace=True)


# In[13]:


df.head()


# ## Pandas profiling

# In[14]:


profile = ProfileReport(df, title="Du Vert près de Chez moi")
profile


# En analysant la corrélation V de Cramer, on peut déduire une forte corrélation entre les colonnes:
# 
#    `[Proposition non réalisable]`  & `[Etat de la proposition]` : 0.7-0.8
# 
#    `[Avancement du projet]` & `[Etat de la proposition]` : 0.7-0.8
#  

# In[15]:


df['Etat de la proposition'].value_counts().to_frame()


# # **<span style="color:#37a871">Groupage des stats dans la même colonne ['Avancement du projet']</span>**
# 
# Le groupage nous permettera de mieux visualiser le status des propositions, si elles ont été retenues ou pas, et le cas échéant l'état d'avancement.

# In[16]:


df.loc[df['Proposition non réalisable'] != '-', ['Avancement du projet']] = "non retenu"


# ### Propositions non réalisables techniquement

# In[17]:


#le dataframe dfnrt contiendra toutes les Propositions non réalisables techniquement
dfnrt= df[df['Etat de la proposition'] == 'Propositions non réalisables techniquement']


# In[18]:


dfnrt


# In[19]:


print(f"La base fait {dfnrt.shape[0]} lignes et {dfnrt.shape[1]} colonnes")


# #### Relation & statistiques entre les modalités de la colonne ['Etat de la proposition'] & ['Avancement du projet'] <span style="color:red"> avant processing </span>

# In[20]:


df.loc[:,['Etat de la proposition','Avancement du projet']].value_counts().to_frame()


# In[21]:


#le dataframe dfr contiendra toutes les Propositions réalisables mais pas forcément retenues
dfr = df.drop(dfnrt.index)
dfr


# In[22]:


print(f"La base fait {dfr.shape[0]} lignes et {dfr.shape[1]} colonnes")


# #### Les modalités de la colonne ['Etat de la proposition'] correspondant à "-" dans la colonne ['Proposition non réalisable] :

# In[23]:


pd.DataFrame(dfr[dfr['Proposition non réalisable'] == '-']["Etat de la proposition"].value_counts())


# #### Groupage des propositions non réalisables/non retenues dans le groupe "non retenu" de la colonne ['Avancement du projet']

# In[24]:


df.loc[df["Etat de la proposition"].str.contains("Propositions non réalisables techniquement|non retenue dans les 200"),['Avancement du projet']] = 'non retenu'


# #### Groupage des modalités "Réalisé" et "Programmé" sous le groupe "retenu" dans la colonne ['Avancement du projet']

# In[25]:


#Groupage des propositions réalisées/programmées en un grand groupe : "retenu"
df.loc[df["Avancement du projet"].str.contains("Réalisé"),['Avancement du projet']] = 'retenu'


# In[26]:


df.loc[df["Avancement du projet"].str.contains("Programmé"),['Avancement du projet']] = 'retenu'


# In[27]:


retenu = df[df['Avancement du projet'].eq('retenu')]


# #### Groupage des modalités "en cours d'études/De substitution/De validation" sous le groupe "en cours de validation" dans la colonne ['Avancement du projet']

# In[28]:


#Groupage des propositions En cours d'études/De substitution/De validation en un grand groupe : "en cours de validation"
df.loc[df["Avancement du projet"].str.contains("En cours d'études"),['Avancement du projet']] = 'en cours de validation'


# In[29]:


df.loc[df["Avancement du projet"].str.contains("En cours de substitution / En cours de validation"),['Avancement du projet']] = 'en cours de validation'


# In[30]:


encours = df[df['Avancement du projet'].eq('en cours de validation')]


# In[31]:


df['Avancement du projet'].value_counts().to_frame()


# #### 3 groupes distincts à partir desquels on peut dresser les stats par Arrondissements

# #### Relation & statistiques entre les modalités de la colonne ['Etat de la proposition'] & ['Avancement du projet'] <span style="color:red"> après processing </span>

# In[32]:


df.loc[:,['Etat de la proposition','Avancement du projet']].value_counts().to_frame()


# # **<span style="color:#37a871">Stats par Arrondissement de toutes les propositions</span>**

# In[33]:


#Indexation par Arrondissement
df=df.set_index('Arrondissement')


# In[34]:


#Stats pourcentages par Arrondissements
df['Avancement du projet'].groupby(df.index).value_counts(normalize=True).mul(100).round(decimals=0).to_frame()


# In[35]:


arr= list(range(75001,75021))
lis = []
for i in arr: 
    lis.append(pd.DataFrame(df[df.index == i]['Avancement du projet'].value_counts()).loc['retenu'][0])
dd = pd.concat([pd.DataFrame(df.index.value_counts()),pd.concat([pd.DataFrame([df.loc[df.index ==i,['Avancement du projet']].value_counts().values[0]], index=[i], columns=['non retenue']) for i in range(75001,75021)])],axis=1)
dd = pd.concat([dd,pd.DataFrame(lis,index=arr, columns=['retenue'])],axis=1)
dd['En cours'] = dd['Arrondissement'] - dd['non retenue'] - dd['retenue']
dd = dd.reset_index()
dd.columns = ['Arrondissement','Propositions','non retenue','retenue','en cours']
dd['retenue + non retenue'] = dd['non retenue'] + dd['retenue']
dd


# In[36]:


sns.set_theme(style="whitegrid")

f, ax = plt.subplots(figsize=(20, 6))

sns.set_color_codes("bright")
sns.barplot(x="Arrondissement", y="Propositions", data=dd,
            label="en cours", color="g")


sns.set_color_codes("bright")
sns.barplot(x="Arrondissement", y="retenue + non retenue", data=dd,
            label="retenue", color="orange")


sns.set_color_codes("bright")
sns.barplot(x="Arrondissement", y="non retenue", data=dd,
            label="non retenue", color='b')


ax.legend(ncol=2, loc="upper left", frameon=True)
ax.set(xlim=(-0.5, 20), ylabel="",
       xlabel="Stats totales par arrondissement")
sns.despine(left=True, bottom=True)


# In[37]:


#Pie chart de toutes les propositions présentes dans la base de données
df['Avancement du projet'].value_counts().plot(kind='pie',ylabel='', title= 'Stats totales des propositions', autopct='%1.f%%', legend = False, figsize=(12,6), fontsize=12,explode = [0, 0.2, 0.2]);


# ## **<span style="color:#37a871">Stats des propositions retenues par Arrondissement</span>**

# In[38]:


# data to plot
n_groups = 20
retenue = dd['retenue']
encours = dd['en cours']

# create plot
fig, ax = plt.subplots(figsize=(26,8), dpi=70)
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, retenue, bar_width,
alpha=opacity,
color='b',
label='Retenue')

rects2 = plt.bar(index + bar_width, encours, bar_width,
alpha=opacity,
color='g',
label='En cours de validation')

plt.xlabel('Arrondissement', fontsize= 25)
plt.ylabel('Nombre de propositions', fontsize=25)
plt.title('Propositions par Arrondissement Retenue et en cours de validation', fontsize= 30)
plt.xticks(index + bar_width, list(range(75001,75021)),fontsize= 18)
plt.legend(fontsize=20)

plt.tight_layout()
plt.show();


# ### Types de végétalisation

# In[39]:


warnings.filterwarnings("ignore", 'This pattern has match groups' )


# In[40]:


#Toutes les propositions non retenues
nonretenu= df[df['Avancement du projet'] == 'non retenu'].copy(deep=True)
nonretenu


# In[41]:


retenu=retenu.copy(deep=True)

retenu


# In[42]:


encours = df[df['Avancement du projet'].eq('en cours de validation')]
encours=encours.copy(deep=True)
encours


# In[43]:


nonretenu['Type de végétalisation'] = nonretenu['Type de végétalisation'].str.lower()


# In[44]:


retenu['Type de végétalisation'] = retenu['Type de végétalisation'].str.lower()


# In[45]:


encours['Type de végétalisation'] = encours['Type de végétalisation'].str.lower()


# In[46]:


#Type de végétalisation des projets retenus
retenu['Type de végétalisation'].value_counts().to_frame()


# In[47]:


#Type de végétalisation des projets en cours
encours['Type de végétalisation'].value_counts().to_frame()


# In[48]:


#Type de végétalisation des projets non retenus
nonretenu['Type de végétalisation'].value_counts().to_frame()


# In[49]:


#Vérification des NaN
retenu['Type de végétalisation'].isnull().sum()


# In[50]:


encours['Type de végétalisation'].isnull().sum()


# In[51]:


nonretenu['Type de végétalisation'].isnull().sum()


# In[52]:


#Inspection des lignes en question
pd.options.display.max_colwidth = 300
nonretenu[nonretenu['Type de végétalisation'].isnull()]


# In[53]:


nonretenu['Type de végétalisation']=nonretenu['Type de végétalisation'].replace(np.nan, 'Autres', regex=True)


# #### Groupage & harmonisation des modalités

# In[54]:


#Modalité "Pots/Bacs/Jardinière"
retenu.loc[retenu['Type de végétalisation'].str.contains(r'(^.*pot.*$)|(^.*fleur.*$)|(^.*jardinière.*$)'), ['Type de végétalisation']] = "Pots/Bacs/Jardinière"

encours.loc[encours['Type de végétalisation'].str.contains(r'(^.*pot.*$)|(^.*fleur.*$)|(^.*jardinière.*$)'), ['Type de végétalisation']] = "Pots/Bacs/Jardinière"

nonretenu.loc[nonretenu['Type de végétalisation'].str.contains(r'(^.*pot.*$)|(^.*fleur.*$)|(^.*jardinière.*$)'), ['Type de végétalisation']] = "Pots/Bacs/Jardinière"


# In[55]:


#Modalité "Mur végétlisé"
retenu.loc[retenu['Type de végétalisation'].str.contains(r'(^.*mur.*$)'), ['Type de végétalisation']] = "Mur végétalisé"

encours.loc[encours['Type de végétalisation'].str.contains(r'(^.*mur.*$)'), ['Type de végétalisation']] = "Mur végétalisé"

nonretenu.loc[nonretenu['Type de végétalisation'].str.contains(r'(^.*mur.*$)'), ['Type de végétalisation']] = "Mur végétalisé"


# In[56]:


#Modalité "Plantations d'arbres"
retenu.loc[retenu['Type de végétalisation'].str.contains(r'(^.*arbre.*$)' or r'(^.*plantation.*$)'or r'(^.*comestible*.$)'), ['Type de végétalisation']] = "Plantations d'Arbres"

encours.loc[encours['Type de végétalisation'].str.contains(r'(^.*arbre.*$)' or r'(^.*plantation.*$)'or r'(^.*comestible*.$)'), ['Type de végétalisation']] = "Plantations d'Arbres"

nonretenu.loc[nonretenu['Type de végétalisation'].str.contains(r'(^.*arbre.*$)' or r'(^.*plantation.*$)'or r'(^.*comestible*.$)'), ['Type de végétalisation']] = "Plantations d'Arbres"


# In[57]:


#Modalité "Végétalisations"
retenu.loc[retenu['Type de végétalisation'].str.contains(r'(^.*verger.*$)|(^.*plante.*$)|(^.*esplanade.*$)|(^.*végétalisation.*$)'), ['Type de végétalisation']] = "Végétalisations"

encours.loc[encours['Type de végétalisation'].str.contains(r'(^.*verger.*$)|(^.*plante.*$)|(^.*esplanade.*$)|(^.*végétalisation.*$)'), ['Type de végétalisation']] = "Végétalisations"

nonretenu.loc[nonretenu['Type de végétalisation'].str.contains(r'(^.*verger.*$)|(^.*plante.*$)|(^.*esplanade.*$)|(^.*végétalisation.*$)'), ['Type de végétalisation']] = "Végétalisations"


# In[58]:


#Modalité "Autres"
retenu.loc[retenu['Type de végétalisation'].str.contains(r'(^.*autre*.$)|(^.*proposition*.$)'), ['Type de végétalisation']] = "Autres"

encours.loc[encours['Type de végétalisation'].str.contains(r'(^.*autre*.$)|(^.*proposition*.$)'), ['Type de végétalisation']] = "Autres"

nonretenu.loc[nonretenu['Type de végétalisation'].str.contains(r'(^.*autre*.$)|(^.*proposition*.$)'), ['Type de végétalisation']] = "Autres"


# In[59]:


retenu['Type de végétalisation'].value_counts().to_frame()


# In[60]:


encours['Type de végétalisation'].value_counts().to_frame()


# In[61]:


nonretenu['Type de végétalisation'].value_counts().to_frame()


# In[62]:


#Type de végétalisation  retenus
fig, ax = plt.subplots(figsize=(12,7), dpi=70)
plt.xticks(rotation=35)
retenu['Type de végétalisation'].hist(ax=ax,alpha=1,rwidth=0.5)
plt.title("Types de végétalisation des propositions retenues");


# In[63]:


#Type de végétalisation  en cours
fig, ax = plt.subplots(figsize=(12,7), dpi=70)
plt.xticks(rotation=35)
encours['Type de végétalisation'].hist(ax=ax,alpha=1,rwidth=0.5)
plt.title("Type de végétalisation  des propositions en cours");


# In[64]:


#Type de végétalisation  non retenues
fig, ax = plt.subplots(figsize=(12,7), dpi=70)
plt.xticks(rotation=35)
nonretenu['Type de végétalisation'].hist(ax=ax,alpha=1,rwidth=0.5)
plt.title("Type de végétalisation  des propositions non retenues");


# ### Stats des causes de refus des propositions

# In[65]:


#Les causes du refus des propositions avec ordre décroissant
nonretenu['Proposition non réalisable'].value_counts(ascending=False).to_frame()


# #### Harmonisation des causes de refus par mots clés

# In[66]:


#Vérification des NaN/Lignes vides
nonretenu["Proposition non réalisable"].isna().sum()


# In[67]:


#Inspection des lignes en question
pd.options.display.max_colwidth = 300
nonretenu[nonretenu['Proposition non réalisable'].isna()]['Commentaitre accompagnant la proposition'].to_frame()


# En lisant les commentaires, aucune classification claire ne peut être attribuée à ces 2 propositions

# In[68]:


nonretenu['Proposition non réalisable']=nonretenu['Proposition non réalisable'].replace(np.nan, 'hors contexte', regex=True)


# In[69]:


#Modalité "Domaine privé"
nonretenu.loc[nonretenu['Proposition non réalisable'].str.contains(r'(^.*public*.$)'), ['Proposition non réalisable']] = "Domaine privé"


# In[70]:


#Modalité "Hors contexte"
nonretenu.loc[nonretenu['Proposition non réalisable'].str.contains(r'(^.*sans rapport*.$)|(^.*grande ampleur*.$)'), ['Proposition non réalisable']] = "Hors contexte"


# In[71]:


#Modalité "Réalisable mais non retenue"
nonretenu.loc[nonretenu['Proposition non réalisable']=='-']['Etat de la proposition'].value_counts()


# In[72]:


nonretenu.loc[nonretenu['Proposition non réalisable'].str.contains("-"), ['Proposition non réalisable']] = "Réalisable mais non retenue"


# In[73]:


#Modalité "Espace inadéquat"
nonretenu.loc[~nonretenu['Proposition non réalisable'].str.contains(r'(Domaine privé)|(hors contexte)|(Réalisable mais non retenue)'), ['Proposition non réalisable']] = "Espace inadéquat"


# In[74]:


nonretenu['Proposition non réalisable'].value_counts().to_frame()


# In[75]:


#Stats des causes de refus par arrondissement
nonretenu['Proposition non réalisable'].groupby(nonretenu.index).value_counts(normalize=True).mul(100).round(decimals=0).to_frame()


# ## **<span style="color:#37a871">Stats des causes de refus par Arrondissement</span>**

# In[76]:


frame = nonretenu['Proposition non réalisable'].groupby(nonretenu.index).value_counts().to_frame().copy(deep=True)
frame.columns = ['Times']
frame.reset_index(inplace=True)
frame = frame.pivot(index='Arrondissement', columns='Proposition non réalisable',values='Times').fillna(0)
frame.reset_index(inplace=True)
frame['Dom. Pr + Esp Inad'] = frame['Domaine privé'] + frame['Espace inadéquat']
frame['Dom. Pr + Esp Inad + Réal Non Ret'] = frame['Dom. Pr + Esp Inad'] + frame['Réalisable mais non retenue']
frame['Total'] = frame['Dom. Pr + Esp Inad + Réal Non Ret'] + frame['hors contexte']


# In[77]:


sns.set_theme(style="whitegrid")

f, ax = plt.subplots(figsize=(20, 6))

sns.set_color_codes("bright")
sns.barplot(x="Arrondissement", y="Total", data=frame,
            label="hors contexte", color="g")


sns.set_color_codes("bright")
sns.barplot(x="Arrondissement", y="Dom. Pr + Esp Inad + Réal Non Ret", data=frame,
            label="Réalisable mais non retenue", color="orange")


sns.set_color_codes("bright")
sns.barplot(x="Arrondissement", y="Dom. Pr + Esp Inad", data=frame,
            label="Espace inadéquat", color='b')

sns.set_color_codes("bright")
sns.barplot(x="Arrondissement", y="Domaine privé", data=frame,
            label="Domaine privé", color='r')


ax.legend(ncol=2, loc="upper left", frameon=True)
ax.set(xlim=(-0.5, 20), ylabel="",
       xlabel="Causes de refus par Arrondissement")
sns.despine(left=True, bottom=True)


# ## <h1><center><span style="color:RED"> Model prediction </span> </center></h1>

# In[78]:


retenu.set_index(['Arrondissement'], inplace=True)


# In[79]:


df = pd.concat([retenu, encours, nonretenu], axis=0)


# In[80]:


#Réduction des types de modalités dans la colonne Target
df.loc[df['Avancement du projet'].str.contains('en cours'), ['Avancement du projet']] = "retenu"


# [![11.png](https://i.postimg.cc/DZ28mpZ6/11.png)](https://postimg.cc/sBNVNc5G)

# En comparant la corrélation V de Cramer dans les 2 DF:
# 
# La variable `[Type de végétalisation]` présentant une corrélation < 0.4 avec les autres variables, on a donc supprimé cette dernière
# 
# 
# DF initiale: 
# 
#    `[Proposition non réalisable]`  & `[Etat de la proposition]` : 0.7-0.8
#    
#    `[Avancement du projet]` & `[Etat de la proposition]` : 0.7-0.8
# 
# DF modifié: 
# 
#    `[Proposition non réalisable]`  & `[Etat de la proposition]` : 0.9
#    
#    `[Avancement du projet]` & `[Etat de la proposition]` : 0.9
#   
#  

# In[81]:


df.drop(columns=['Geo Point','Type de végétalisation'], axis=1, inplace=True)


# ### Split de la base de données

# In[82]:


print(f"La base fait {df.shape[0]} lignes et {df.shape[1]} colonnes")


# In[83]:


df.columns


# In[84]:


#X étant la dataframe des variables d'entrainement & Y la dataframe de la variable target

X_train = df[['Proposition non réalisable','Etat de la proposition','Commentaitre accompagnant la proposition']]
Y_train = df[['Avancement du projet']]


# ## Le split Train/Validation/Test :
# 
# - train = 53% df 
# - validation = 22% df 
# - test = 25% df

# In[85]:


X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.25, random_state=0)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=0)


# In[86]:


print(f"Les bases X_train & Y_train font {X_train.shape[0]} & {Y_train.shape[0]} lignes et {X_train.shape[1]} & {Y_train.shape[1]} colonnes")

print(f"Les bases X_val & Y_val font {X_val.shape[0]} & {Y_val.shape[0]} lignes et {X_val.shape[1]} & {Y_val.shape[1]} colonnes")

print(f"Les bases X_test & Y_test font {X_test.shape[0]} & {Y_test.shape[0]} lignes et {X_test.shape[1]} & {Y_test.shape[1]} colonnes")


# ## Encoding des données

# In[87]:


label_encoder = preprocessing.LabelEncoder()


# In[88]:


X_train_encod=X_train.apply(label_encoder.fit_transform)
Y_train_encod=Y_train.apply(label_encoder.fit_transform)
X_val_encod=X_val.apply(label_encoder.fit_transform)
Y_val_encod=Y_val.apply(label_encoder.fit_transform)
X_test_encod=X_test.apply(label_encoder.fit_transform)
Y_test_encod=Y_test.apply(label_encoder.fit_transform)


# Le `Label encoding` assigne par défaut 0 /1 suivant l'ordre alphabétique
# 
# `0` = `non retenu`
# 
# `1` = `retenu`

# ### Train

# In[89]:


#Entrainement du modele 
logreg= LogisticRegression()
LogisticRegression(solver='lbfgs', max_iter=10000)
logreg.fit(X_train_encod,Y_train_encod.values.ravel()) #Y_train_encod = nparray


# ### Validation

# In[90]:


#Validation
Y_pred_val=logreg.predict(X_val_encod)


# In[91]:


class_report_v = classification_report(Y_val_encod, Y_pred_val, output_dict=True,target_names=['Non retenu', 'Retenu'],zero_division=1)
pd.DataFrame(class_report_v)


# ### Test

# In[92]:


Y_pred_test=logreg.predict(X_test_encod)


# In[93]:


#Rapport de prédiction du X_test
class_report_t = classification_report(Y_test_encod, Y_pred_test,output_dict=True,target_names=['Non retenu', 'Retenu'], zero_division = 1)
pd.DataFrame(class_report_t)


# `recall` : Nombre de classes trouvées par rapport aux nombres entiers de cette même classe.
# 
# `precision` : Combien de classes ont été correctements classifiées
# 
# `f1-score` : La moyenne harmonique entre precision & recall
# 
# `support` : Le nombre d'occurences de la classe donnée dans le dataframe 

# In[94]:


warnings.filterwarnings("ignore", 'FixedFormatter' )


# In[95]:


#Matrice de confusion normalizée par rapports aux prédictions
cm = confusion_matrix(Y_test_encod, Y_pred_test)
print("La matrice de confusion : \n{}".format(cm))
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
labels = ['non retenu', 'retenu']
plt.title('Matrice de confusion normalizée/predictions', loc='center', fontsize=15)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predictions')
plt.ylabel('Réelles')
plt.show()

