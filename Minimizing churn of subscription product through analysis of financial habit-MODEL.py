#### Importing Libraries ####
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import random

# Loading the dataset
dataset = pd.read_csv('new_churn_data.csv')
dataset

## Data Preparation
user_identifier = dataset['user']
dataset = dataset.drop(columns = ['user'])

## One-Hot Encoding (for our categordical variable to convert them in binary form as python can not read the categorical data)
dataset.housing.value_counts()
dataset = pd.get_dummies(dataset)
dataset.columns
dataset.shape

#We need to avoid dummy variable trap (which is quasi correlated field)
dataset = dataset.drop(columns = ['housing_na', 'payment_type_na', 'zodiac_sign_na'])
dataset.shape

X = dataset.drop(columns = ['churn'])
y = dataset['churn']

## Splitting the Training & Testing Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y, train_size = 0.8, random_state = 0)

#Balacing the trainingset 
y_train.value_counts()

pos_index = y_train[y_train == 1].index
neg_index = y_train[y_train == 0].index

if len(pos_index) > len(neg_index):
    higher = pos_index
    lower = neg_index
else:
    lower = pos_index
    higher = neg_index
    
random.seed(0)
higher = np.random.choice(higher, size = len(lower))
lower = np.asarray(lower)
new_indexes = np.concatenate((lower, higher))

X_train = X_train.loc[new_indexes,]
y_train = y_train[new_indexes]

## Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values

X_train = X_train2
X_test = X_test2


## Fitting Model to the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train, sample_weight = None)

#Predicting the Test Set
y_pred = classifier.predict(X_test)

# Evaluating the Resiles
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)

accuracy_score(y_test, y_pred) # how accurate the data is, how close the prediction is to testing data
precision_score(y_test, y_pred) # true_positive / (true_positive + false_positive)
recall_score(y_test, y_pred) # true_positive / (true_positive + false_negative)
f1_score(y_test, y_pred) # it is similart to accuracy, combination of precision score & recall score.

#Plottign the confusion matrix

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10, 7))
sn.set(font_scale = 1.4)
sn.heatmap(df_cm, annot = True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))

# Applying K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, #classifier is the regression we build earlier
                             X = X_train,
                             y = y_train,
                             cv = 100) # cv represents how many folds we wants
accuracies
accuracies.mean()

# Applying Coefficient
pd.concat([pd.DataFrame(X_train.columns, columns = ["features"]), 
           pd.DataFrame(np.transpose(classifier.coef_), columns = ["coef"])],
axis = 1)

#### Feature Selection ####
# (selecting all those features with wich the same accurac can be attainable, with less features the computation will be 
# be faster and acuire less memories)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Model to Test
classifier = LogisticRegression()
rfe = RFE(classifier, 20) # 20, represents how many feaures we wants to have almost same accuracy
rfe = rfe.fit(X_train, y_train)

# summarize the selection of the attributes
print(rfe.support_)
X_train.columns[rfe.support_]

print(rfe.ranking_)

##################################################################################################
#### now checking the model performance on these dataset only, so for this copying the same code which we used before
# and passing the values in the mode to evaluate the model performance.


## Fitting Model to the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train[X_train.columns[rfe.support_]], y_train, sample_weight = None)

#Predicting the Test Set
y_pred = classifier.predict(X_test[X_test.columns[rfe.support_]])

# Evaluating the Resiles
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)

accuracy_score(y_test, y_pred) # how accurate the data is, how close the prediction is to testing data
precision_score(y_test, y_pred) # true_positive / (true_positive + false_positive)
recall_score(y_test, y_pred) # true_positive / (true_positive + false_negative)
f1_score(y_test, y_pred) # it is similart to accuracy, combination of precision score & recall score.

#Plottign the confusion matrix

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10, 7))
sn.set(font_scale = 1.4)
sn.heatmap(df_cm, annot = True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))

# Analyzing Coefficient (the top 20 extracted features and coefficient attached with them)
pd.concat([pd.DataFrame(X_train.columns[rfe.support_], columns = ["features"]), 
           pd.DataFrame(np.transpose(classifier.coef_), columns = ["coef"])],
axis = 1)


#######################################
########## End of MOdel################

# Formatting FInals Results
final_results = pd.concat([y_test, user_identifier], axis = 1).dropna()
final_results['predicted_churn'] = y_pred
final_results = final_results[['user', 'churn', 'predicted_churn']].reset_index(drop = True)





