#!/usr/bin/env python
# coding: utf-8

# # Dataset 
# The data ste contains 7 columns and 5000 rows

# # Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

get_ipython().run_line_magic('matplotlib', 'inline')


# # Load the data set

# In[2]:


data = pd.read_csv('https://raw.githubusercontent.com/al7alm123/Project_Proposal/main/USA_Housing.csv')
data.head()


# In[3]:


# Print the shape of dataset
print(data.shape)


# In[4]:


#get statistical information about data set
data.describe()


# In[5]:


data.info()


# In[6]:


#check for null data
data.isnull().sum


# In[7]:


data.rename(columns={'Avg. Area Income' : 'income',
                   'Avg. Area House Age':'age',
                   'Avg. Area Number of Rooms':'Number_room',
                   'Avg. Area Number of Bedrooms':'Number_bedroom',
                   'Area Population':'Population',
                   'Price':'Price',
                   'Address':'Address'},inplace=True)
data.columns


# In[8]:


#drop unwanted columns
#data.drop('Address', axis = 1, inplace = False)


# # Exploratory Data Analysis

# In[9]:


cor=data.corr()
cor


# In[10]:


#exploring dataset using plots
sns.pairplot(data)


# In[11]:


# heatmap of correlation among various attributes

sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
plt.subplots(figsize=(10,10))
p = sns.heatmap(cor, annot=True, lw=1.5, fmt='.2f', cmap='coolwarm')
rotxlabel = p.set_xticklabels(p.get_xticklabels(),fontdict={'fontsize':10}, rotation=90)
rotylabel = p.set_yticklabels(p.get_yticklabels(),fontdict={'fontsize':10}, rotation=30)


# In[47]:


plt.figure(figsize=(16,8))
plt.subplot(2,3,1)
sns.boxplot(data['income'])
plt.subplot(2,3,2)
sns.boxplot(data['age'])
plt.subplot(2,3,3)
sns.boxplot(data['Number_room'])
plt.subplot(2,3,4)
sns.boxplot(data['Number_bedroom'])
plt.subplot(2,3,5)
sns.boxplot(data['Population'])


# In[12]:


sns.distplot(data['Price'],kde=True,color='red')


# In[13]:


sns.barplot(data['income'], data['Price'])


# In[14]:


data.columns


# In[15]:


X = data[['income', 'age', 'Number_room', 'Number_bedroom', 'Population']]

y = data['Price']


# # Difine x and y

# In[16]:


print(X)


# In[17]:


print(y)


# #  Split dataset into Training set, Test set
# 
# split our dataset into a training set and testing set using sklearn train_test_split(). the training set will be going to use for training the model and testing set for testing the model. We are creating a split of 20% training data and 80% of the training set.

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Train the model on the traing set

# In[19]:


#import and create sklearn linearmodel LinearRegression object and fit the training dataset in it.


lr = LinearRegression(normalize=True)
lr.fit(X_train, y_train)


# # LinearRegression Model 

# In[20]:


#evaluate the model by checking out its coefficients and how we can interpret them.
print(lr.intercept_)


# In[21]:


print(lr.coef_)


# In[22]:


#coeff_df = pd.DataFrame(lr.coef_, X.columns, columns=['Coefficient'])
#coeff_df

coeff_df = pd.DataFrame(lr.coef_, X.columns, columns = ['Coefficient'])
coeff_df
# shows that increase in 1 unit affect each columns by this much


# # Predict the test set results

# In[23]:


predictions = lr.predict(X_test)


# In[24]:


plt.scatter(y_test,predictions)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')


# In this scatter plot, we see data is in a line form, which means our model has done good predictions.

# In[35]:


sns.distplot((y_test-predictions),bins=50);


# In[25]:


sns.displot(data=(y_test - predictions))


# # Regression Evaluation Metrics

# In[36]:


def cross_val(lr):
    predict = cross_val_score(model, X, y, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE', mae)
    print('MSE', mse)
    print('RMSE', rmse)
    print('R2 square', r2_square)
 
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square


# In[28]:


test_predict = lr.predict(X_test)
train_predict = lr.predict(X_train)


# In[37]:


print("Test Set Evaluation")
print_evaluate(y_test, test_predict)


# In[38]:


print("Train Set Evaluation")
print_evaluate(y_train, train_predict)


# In[ ]:





# In[ ]:




