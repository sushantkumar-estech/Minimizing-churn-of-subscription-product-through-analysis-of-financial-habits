# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 23:36:03 2020

@author: Sushant Kumar
"""

## Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Importing the dataset
dataset = pd.read_csv('churn_data.csv')

#Analyzing the dataset
dataset.head()
dataset.tail()

dataset.describe()
dataset.info()

dataset.columns
## Cleaning the dataset

#Removing Nan
dataset.isna().any()
dataset.isna().sum()
dataset = dataset[pd.notnull(dataset['age'])]
dataset = dataset.drop(columns = ['credit_score', 'rewards_earned'])
dataset.shape

## Data plotting
sns.countplot(dataset['churn'], label='count')
plt.show()

sns.countplot(x = 'churn', data= dataset) #this way also the data can be plot

## Histgram
# dropping the dataset which are not required to plot the histgram / or not needed to plot the histgram
dataset2 = dataset.drop(columns = ['user', 'churn'])

# Histogram
plt.figure(figsize = (40,30))
plt.suptitle('Histogram of Numerical Columns', fontsize =20)
for i in range (1, dataset2.shape[1]+ 1):
  plt.subplot(6,5,i) #size of 6X35 will be sufficient
  f = plt.gca() # cleans up everything and clear the field.
  f.axes.get_yaxis().set_visible(False)
  f.set_title(dataset2.columns.values[i-1])
    
  vals = np.size(dataset2.iloc[:,i-1].unique()) # to decide how many bins will be proper for feature
  
  plt.hist(dataset2.iloc[:,i-1], bins=vals, color = 'b')
# this command is not working plt.tight_layout(rect=[0,0.03,1,0,95])
 
##Pie-Plots
# as we will see pie charts for the binary columns, so we will select those columns only

dataset2 = dataset[['housing', 'is_referred', 'app_downloaded', 
                    'web_user', 'app_web_user', 'ios_user',
                    'android_user', 'registered_phones', 'payment_type',
                    'waiting_4_loan', 'cancelled_loan',
                    'received_loan', 'rejected_loan', 'zodiac_sign',
                    'left_for_two_month_plus', 'left_for_one_month', 'is_referred']]  

plt.figure(figsize = (40,20))
plt.suptitle('Pie chart distribution', fontsize =20)
for i in range (1, dataset2.shape[1]+ 1):
  plt.subplot(6,5,i) #size of 6X35 will be sufficient
  f = plt.gca() # cleans up everything and clear the field.
  f.axes.get_yaxis().set_visible(False)
  f.set_title(dataset2.columns.values[i-1])
   
  values = dataset2.iloc[:, i-1].value_counts(normalize = True).values
  index = dataset2.iloc[:, i-1].value_counts(normalize = True).index
  
  plt.pie(values, labels = index, autopct= '%1.1f%%')
  plt.axis('equal')


#now checking the distribution of churn for the variable which have very samll subset
#this is to check the dataset where they are uneven, and how euneven are they
dataset[dataset2.waiting_4_loan == 1].churn.value_counts()
dataset[dataset2.waiting_4_loan == 1].churn.value_counts()
dataset[dataset2.received_loan == 1].churn.value_counts()
dataset[dataset2.waiting_4_loan == 1].churn.value_counts()
dataset[dataset2.left_for_one_month == 1].churn.value_counts()

# Paiplot
sns.pairplot(dataset, palette=True) # Generating the pairplot to see the dependencies among the dataset
sns.pairplot(dataset, hue='churn') # Needs to check this code further for pairplot
## Exploring Uneven Features
# as this requrie only numerical dataset, so we are removing categorical data

dataset_num = dataset.drop(columns = ['churn', 'user', 'housing', #removing categorical dataset
                        'payment_type', 'zodiac_sign'])
dataset_num.corrwith(dataset.churn).plot.bar(figsize = (20, 10), title = 'Correlation with  Response Variable', 
                     fontsize = 15, rot = 45, grid = True, colors = 'c')


# Building the correlation matrxi
# Correlation Matrix (every filed correlation with each other) we dont want features to be dependednt on any other.
# For machine learning model, ist important to have feaures independent from any other featurs.

# Correlation Matrix
sns.set(style="white") # this built the background

#Compute the correlation matrix
corr = dataset.drop(columns = ['user', 'churn']).corr() # the key of correlation, making 2D array to join all the features to each others

#Generate the mask for upper traingle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Setup the matplotlib figure
f, ax = plt.subplots(figsize = (40,20))
f.suptitle("Correlation Matrix", fontsize = 40)

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
           square=True, linewidth=.5, cbar_kws={"shrink": .5})

# Correlation Matrix witht the numercial dataset
sns.set(style="white") # this built the background

#Compute the correlation matrix
corr = dataset_num.corr() # the key of correlation, making 2D array to join all the features to each others

#Generate the mask for upper traingle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Setup the matplotlib figure
f, ax = plt.subplots(figsize = (40,20))
f.suptitle("Correlation Matrix", fontsize = 40)

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
           square=True, linewidth=.5, cbar_kws={"shrink": .5})

#now removing the coorelated datset as we visualize and decided from the correlation matrix
dataset = dataset.drop(columns = ['app_web_user'])
dataset

#now saving the clean data for further work
dataset.to_csv('new_churn_data.csv', index = False) #Finalize of the EDA process

#Data preprocessing of Model builiding process
































