#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA
import sklearn.preprocessing as skl
from sklearn.cluster import KMeans


# In[2]:


data = pd.read_csv("Data_problem_1.csv")
data.head()


# In[3]:


data.tail()


# In[4]:


data.info()


# In[5]:


data.shape


# In[6]:


#data cleansing 
#finding the missing values 

missing_vals = data.isnull().sum()
print("missing values :\n" ,missing_vals)


# In[7]:


#finding unique values 

unique_vals = data.nunique()
print("unique values:\n",unique_vals)


# In[8]:


data.describe()


# In[9]:


#imputing missing values 
#since credit_ limit and min payments is a float value and not a frequency , we impute it with the median

data['CREDIT_LIMIT'].fillna(data['CREDIT_LIMIT'].median() , inplace = True)

data['MINIMUM_PAYMENTS'].fillna(data['MINIMUM_PAYMENTS'].median() , inplace = True)


# In[10]:


print(data.isnull().sum())  #all null values imputed 


# In[11]:


#KPIs


# In[12]:


#1. Monthly average purchase and cash advance amount

#from the dataset desription we know that , 'PURCHASES' is in float64 and 'TENURE' is in int64
#converting int values to float 

data['TENURE'] = data['TENURE'].astype(float)
data.info()



# In[13]:


data['MONTHLY_AVG_PUR'] = data['PURCHASES']/data['TENURE']
print("MONTHLY_AVG_PUR:\n")
data['MONTHLY_AVG_PUR'].head()


# In[14]:


#monthly cash advance amount

data['MONTHLY_CASH_ADV']=data['CASH_ADVANCE']/data['TENURE']
print("MONTHLY_CASH_ADV:\n")
data['MONTHLY_CASH_ADV'].head()


# # 2- Purchases by type (one-off, installments)

# In[15]:


data.loc[: , ['ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES']]


# In[16]:


#in this case we have 4 types 
#1) only oneoff
#2) only installment
#3) both oneoff and installment 
#4) neither oneoff or installment 

def purchase_type(data):
    if (data['ONEOFF_PURCHASES']>0 and data['INSTALLMENTS_PURCHASES']==0):
        return'oneoff'
    
    if (data['ONEOFF_PURCHASES']==0 and data['INSTALLMENTS_PURCHASES']>0):
        return'installment'
        
    if (data['ONEOFF_PURCHASES']>0 and data['INSTALLMENTS_PURCHASES']>0):
        return 'both oneoff and installment'
        
    if (data['ONEOFF_PURCHASES']==0 and data['INSTALLMENTS_PURCHASES']==0):
        return'neither oneoff or installment'
    
data['PURCHASE_TYPE'] = data.apply(purchase_type, axis = 1 )

data['PURCHASE_TYPE'].head()


# In[17]:


data['PURCHASE_TYPE'].value_counts()


# # 3 - Average amount per purchase and cash advance transaction

# In[18]:


data.loc[:,['CASH_ADVANCE_TRX','PURCHASES_TRX']]
         


# In[19]:


def average_vals (data):
    av = (data['CASH_ADVANCE_TRX']+data['PURCHASES_TRX'])/2
    return av

data['AVG_PURCHASE_CASH_ADV'] = data.apply(average_vals ,axis = 1 )

data['AVG_PURCHASE_CASH_ADV'].head


# # 4 - Limit usage (balance to credit limit ratio)
# 

# In[20]:


data['LIMIT_USAGE'] = data.apply(lambda x: x['BALANCE']/x['CREDIT_LIMIT'], axis = 1 )
data['LIMIT_USAGE'].head()


# # 5 -Payments to minimum payments ratio 

# In[21]:


data.isnull().sum()


# In[22]:


data['PAY_TO_MIN_PAY_RATIO'] = data.apply(lambda x: x['PAYMENTS']/x['MINIMUM_PAYMENTS'], axis = 1 )
data['PAY_TO_MIN_PAY_RATIO'].head()


# # Advanced reporting: Use the derived KPIs to gain insight on the customer profiles.

# In[23]:


#1 -Average cash advance taken by customers of different Purchase type


df =data.groupby('PURCHASE_TYPE')

df.first()


# In[24]:


def my_func(data):
        
    return np.mean(data['MONTHLY_CASH_ADV'])

df.apply(my_func)

X = data.PURCHASE_TYPE.unique()
print(X)


# In[25]:


pd.DataFrame.plot(df.apply(my_func)).barh()
# plt.barh(X ,df.apply(my_func))
# plt.title('Average cash advance taken by customers of different Purchase type ')


# # There fore  customers with neither one off or installment purchase type take a higher cash advance 

# In[26]:


#2- to find out which purchase type customers make more payments 


# In[27]:


def my_func2(data):
    return np.mean(data['PAY_TO_MIN_PAY_RATIO'])

df.apply(my_func2)


# In[28]:


print(X)


# In[29]:


pd.DataFrame.plot(df.apply(my_func2)).barh()
plt.title('customers with the one off purchase type take a higher cash advance')


# # Therefore customers with purchase type neither oneoff or installment make more of theri payments 

# # Identification of the relationships/ affinities between services.

# In[30]:


#relationship between payments and minimum payments 


# In[31]:


#Applying PCA as data reduction technique for variable reduction and 
#KMeans for clustering algorithm to reveal the behavioral segments of credit card holders


# In[50]:



features = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']
print(features)


# In[51]:


# list_of_column_names = list(df.columns)

x = data.loc[:, features].values
print(x)
l = len(x)
# Separating out the target
y = data.loc[:,['PURCHASE_TYPE']].values
# print(y
# Standardizing the features
x = skl.StandardScaler().fit_transform(x)
np.round(x,6)


# In[52]:


print(y)


# In[53]:


x.shape


# In[55]:


init_pca=PCA(n_components=17)
cur_pca=init_pca.fit(x)


# In[56]:


#variance expalined at 17 components (ideally 1)

sum(cur_pca.explained_variance_ratio_)


# In[57]:


var_ratio={}
for n in range(2,18):
    pca=PCA(n_components=n)
    cur_pca=pca.fit(x)
    var_ratio[n]=sum(cur_pca.explained_variance_ratio_)


# In[48]:


var_ratio


# In[58]:


# pca = PCA().fit(digits.data)
plt.plot(np.cumsum(cur_pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# In[77]:


#from the baove info we need to take 10 components for pca


# In[78]:


pca=PCA(n_components=10)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents , columns = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10'])
principalDf.head()


# In[79]:


#now , we come to the clustering step 

n_clusters=30
inertias=[]
for i in range(1,n_clusters):
    kmeans= KMeans(init = 'k-means++', n_clusters = i)
    kmeans.fit(principalDf)
    inertias.append(kmeans.inertia_) 


# In[80]:


plt.figure()
plt.plot(inertias,marker ='o')
plt.show


# In[105]:


#therefore , lets have number of clusters as 5 acccoring to the elbow curve

kmeans= KMeans(init = 'k-means++', n_clusters = 5 ,n_init= 10 , max_iter = 100,  )
label = kmeans.fit_predict(principalDf)
labels=kmeans.labels_
print(label)


# In[103]:


data_clusters = pd.concat([principalDf, pd.DataFrame({'cluster':labels})], axis=1)
print(data_clusters)


# In[104]:


data_clusters.cluster.unique()


# In[110]:



plt.figure(figsize=(12,8))

# plots a scatter plot to view the clusters
plt.scatter(x=data['PURCHASES'], y=data['PAYMENTS'], c=labels, s=5, cmap='rainbow', alpha=0.75)

# sets the plot features
plt.xlabel('Purchases')
plt.ylabel('Payments')
plt.title("Payments by Purchases", fontsize=14)

# displays the plot
plt.show()


# In[ ]:




