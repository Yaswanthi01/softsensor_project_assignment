#!/usr/bin/env python
# coding: utf-8

# In[86]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA
import sklearn.preprocessing as skl
import os
from sklearn.cluster import KMeans
import glob
import seaborn as sns


# In[2]:


# data = pd.read_csv("task_2_data.csv")

# data1 = pd.read_csv("task_2_data_1.csv")
# data2 = pd.read_csv("task_2_data_2.csv")
# # data3 = pd.read_csv("task_2_data_3.csv")
# # data4 = pd.read_csv("task_2_data_4.csv")
# # data5 = pd.read_csv("task_2_data_5.csv")
# # data6 = pd.read_csv("task_2_data_6.csv")
# # data7 = pd.read_csv("task_2_data_7.csv")
# # data8 = pd.read_csv("task_2_data_8.csv")
# # data9 = pd.read_csv("task_2_data_9.csv")
# # data10 = pd.read_csv("task_2_data_10.csv")
# # data11 = pd.read_csv("task_2_data_11.csv")
# # data12 = pd.read_csv("task_2_data_12.csv")
# # data.head()


# In[3]:


files = os.path.join("task_2_data*.csv")


# In[4]:


files  = glob.glob(files)


# In[5]:


li =[]
for filename in files:
    frame = pd.read_csv(filename, index_col=None, header=0 ,low_memory=False)
    li.append(frame)
    print(filename)

# df = pd.concat(li, axis=0, ignore_index=True)


# In[6]:


type(li)


# In[7]:


df = pd.concat(li, axis=0, ignore_index=True)


# In[8]:


df.head()


# In[9]:


df.info()


# In[11]:


df.shape


# In[12]:


#data cleansing 
#finding the missing values 

missing_vals = df.isnull().sum()
print("missing values :\n" ,missing_vals)


# In[13]:


#finding unique values 

unique_vals = df.nunique()
print("unique values:\n",unique_vals)


# In[14]:


df.describe()


# In[15]:


print(df['MORTGAGE_INTEREST_PROCEED'])


# In[17]:


print(df['UTILITIES_PROCEED'])


# In[18]:


#dropping clumns with most missing values 

df.drop(['FranchiseName','MORTGAGE_INTEREST_PROCEED','RENT_PROCEED','REFINANCE_EIDL_PROCEED','HEALTH_CARE_PROCEED','DEBT_INTEREST_PROCEED','UTILITIES_PROCEED','NonProfit'], axis = 1)


# In[19]:


df.dropna( axis=0, how="any",  subset=['BorrowerState','OriginatingLenderState','ServicingLenderState'], inplace=True)


# In[20]:


df.shape


# In[21]:


missing_vals = df.isnull().sum()
print("missing values :\n" ,missing_vals)


# In[22]:


#dropping clumns with most missing values 

df.drop(['FranchiseName','MORTGAGE_INTEREST_PROCEED','RENT_PROCEED','REFINANCE_EIDL_PROCEED','HEALTH_CARE_PROCEED','DEBT_INTEREST_PROCEED','UTILITIES_PROCEED','NonProfit'], axis = 1,inplace = True)


# In[23]:


missing_vals = df.isnull().sum()
print("missing values :\n" ,missing_vals)


# In[25]:


df['PAYROLL_PROCEED'].fillna(df['PAYROLL_PROCEED'].median() , inplace = True)


# In[26]:


#dropping columns with most missing values 

df.drop(['ForgivenessDate'], axis = 1,inplace = True)
df['ForgivenessAmount'].fillna(df['ForgivenessAmount'].median() , inplace = True)


# In[27]:


missing_vals = df.isnull().sum()
print("missing values :\n" ,missing_vals)


# In[28]:


df.dropna( axis=0, how="any",  subset=['BusinessType','ProjectState','BusinessAgeDescription','LMIIndicator'], inplace=True)
df.drop(['BorrowerZip','ProjectZip','ProjectCountyName','BorrowerAddress','BorrowerName','CD'], axis = 1,inplace = True)
df['BusinessType'].fillna(df['BusinessType'].mode() , inplace = True)
df['ProjectCity'].fillna(df['ProjectCity'].mode() , inplace = True)
df['BorrowerCity'].fillna(df['BorrowerCity'].mode() , inplace = True)
df['UndisbursedAmount'].fillna(df['UndisbursedAmount'].median() , inplace = True)


# In[29]:


missing_vals = df.isnull().sum()
print("missing values :\n" ,missing_vals)


# In[30]:





# In[31]:


missing_vals = df.isnull().sum()
print("missing values :\n" ,missing_vals)


# In[32]:


df.dropna( axis=0, how="any",  subset=['ProjectCity','BorrowerCity','JobsReported'], inplace=True)

df['NAICSCode'].fillna(df['NAICSCode'].mode() , inplace = True)


# In[33]:


missing_vals = df.isnull().sum()
print("missing values :\n" ,missing_vals)


# In[34]:


df.dropna( axis=0, how="any",  subset=['NAICSCode'], inplace=True)


# In[35]:


missing_vals = df.isnull().sum()
print("missing values :\n" ,missing_vals)


# In[36]:


#stateavg vs national avg


df['nat_avg']=df['InitialApprovalAmount'].mean()


# In[40]:


df.BorrowerState.unique()


# In[44]:


df['state_avg'] = df.groupby(['BorrowerState'])['InitialApprovalAmount'].transform('mean')


# In[45]:


print(df['state_avg'])


# In[50]:


type(df['state_avg'])


# In[62]:


x = list(df.BorrowerState.unique())
y = list(df.state_avg.unique())
y1 = list(df['nat_avg'].unique())*57
x1 = list(df.BorrowerState.unique())
# print(y)
plt.xlabel('states')

plt.ylabel('loan amount')
  
plt.plot(x, y, label = "state average")
plt.plot(x1 , y1, label = "national average" )


# In[63]:


#Average loan for a particular city.

df.BorrowerCity.unique()


# In[64]:


df['city_avg'] = df.groupby(['BorrowerCity'])['InitialApprovalAmount'].transform('mean')


# In[65]:


print(df.groupby(['BorrowerCity'])['InitialApprovalAmount'].mean())


# In[79]:


#Loan Amount grouped by Gender

print(df.groupby(['Gender'])['InitialApprovalAmount'].mean())
df['loan_gender'] = df.groupby(['Gender'])['InitialApprovalAmount'].transform('mean')


# In[91]:


# print((df.groupby(['Gender'])['InitialApprovalAmount'].mean()).size())

result =df.groupby(['Gender'])['loan_gender'].size()

print(result)



# In[90]:


sns.barplot(x = result.index, y = result.values)


# In[92]:


#Highest loan lender in each city.

print(df.groupby(['OriginatingLenderCity'])['OriginatingLender'].max())


# In[93]:


#avg loan taken by veteran vs non-vetrans

print(df.groupby(['Veteran'])['InitialApprovalAmount'].mean())


# In[94]:


#number of vetrans and non-veterans taking loan 
print(df.groupby(['Veteran'])['InitialApprovalAmount'].size())


# In[95]:


#no of loans based on ethnicity 

print(df.groupby(['Ethnicity'])['InitialApprovalAmount'].size())


# In[96]:


# no of cases where loan is disbursed but not Paid In Full or Charged Off

df['LoanStatusDate'].isnull().sum()


# In[99]:


df['LoanStatusDate'].notnull().sum()


# In[100]:


#ratio for the above case :

rat = df['LoanStatusDate'].isnull().sum()/df['LoanStatusDate'].notnull().sum()
print(rat)


# In[ ]:




