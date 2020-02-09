# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:04:15 2020

@author: shive
"""

#reference: https://www.kaggle.com/c/titanic/notebooks

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
import os

os.chdir("C:\\Users\\shive\\Desktop\\MS-BAIM\\Courses\\competition\\titanic")

train= pd.read_csv("train.csv")
test= pd.read_csv("test.csv")
pass_id= test["PassengerId"]
pass_id= pd.DataFrame(pass_id)
df_y= train.iloc[:, 1].values
train.describe()
train.info()
working_train= train.iloc[:, [2,4,5,6,7,9]]
working_test= test.iloc[:, [1,3,4,5,6,8]]

#data exploration
sns.heatmap(working_train.isnull(), yticklabels=False, cbar=True, cmap='Dark2')

sns.set_style()
sns.countplot(x="Survived", data= train)
plt.title("survival histogram")
plt.show()

sns.countplot(x="Survived", hue="Sex", data= train)
plt.title("Gender Vs Survival")
plt.show()

sns.countplot(x="Survived", hue="Pclass", data= train)
plt.title("Gender Vs Survival")
plt.show()

sns.distplot(train["Age"].dropna(), kde= False, bins=10)

plt.hist(train["Fare"])

#summary of null value
train.isnull().sum()
test.isnull().sum()
working_train.isnull().sum()
working_test.isnull().sum()

x_train= working_train.iloc[:,:].values

#imputing missing values in train data
from sklearn.impute import SimpleImputer
imputer= SimpleImputer(missing_values=np.nan, strategy="median")
imputer= imputer.fit(working_train[["Age"]])
working_train[["Age"]]= imputer.transform(working_train[["Age"]])

working_train.isnull().sum()

#imputing missing values in test data
for dataset in working_test:
    working_test["Age"].fillna(working_test["Age"].median(), inplace=True)
    working_test["Fare"].fillna(working_test["Fare"].median(), inplace=True)

working_test.isnull().sum()

#dummies on train data
sex= pd.get_dummies(working_train["Sex"], drop_first= True)
p_class= pd.get_dummies(working_train["Pclass"], drop_first= True)
working_train.drop(["Sex", "Pclass"], axis=1, inplace= True)
working_train= pd.concat([working_train, sex, p_class], axis=1)
working_train= working_train.rename(columns= {2: "PClass2", 3: "PClass3"})

#dummies on test data
sex_t= pd.get_dummies(working_test["Sex"], drop_first= True)
p_class_t= pd.get_dummies(working_test["Pclass"], drop_first= True)
working_test.drop(["Sex", "Pclass"], axis=1, inplace= True)
working_test= pd.concat([working_test, sex_t, p_class_t], axis=1)
working_test= working_test.rename(columns= {2: "PClass2", 3: "PClass3"})

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_test = train_test_split(working_train, df_y, test_size= 0.2, random_state=0) 

#feature scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train= sc.fit_transform(X_train)
x_valid= sc.transform(X_valid)
x_test= sc.transform(working_test)

#fitting logistic regression
from sklearn.linear_model import LogisticRegression
lr_classifier= LogisticRegression(random_state=0)
lr_classifier.fit(x_train, y_train)

lr_y_pred= lr_classifier.predict(x_valid)
lr_y_test= lr_classifier.predict(x_test)
lr_y_test= pd.DataFrame(lr_y_test)

#K-fold cross validation
from sklearn.model_selection import cross_val_score
lr_accuracies= cross_val_score(estimator= lr_classifier, 
                                X= x_train, 
                                y= y_train,
                                cv=10)
lr_acc= lr_accuracies.mean()

#preparing Kaggle submission file
lr_submission= pd.concat([pass_id, lr_y_test], axis=1)
lr_submission.to_csv("lr_submission.csv")



#fitting KNN model
from sklearn.neighbors import KNeighborsClassifier
knn_classifier= KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
knn_classifier.fit(x_train, y_train)

#predicting
knn_y_pred= knn_classifier.predict(x_valid)
knn_y_test= knn_classifier.predict(x_test)
knn_y_test= pd.DataFrame(knn_y_test)

#K-fold cross validation
from sklearn.model_selection import cross_val_score
knn_accuracies= cross_val_score(estimator= knn_classifier, 
                                X= x_train, 
                                y= y_train,
                                cv=10)
knn_acc= knn_accuracies.mean()

#preparing Kaggle submission file
knn_submission= pd.concat([pass_id, knn_y_test], axis=1)
knn_submission.to_csv("knn_submission.csv")

#fitting SVM Model
from sklearn.svm import SVC
svm_classifier= SVC(kernel= "rbf", random_state=0, C=1, gamma= 0.3)
svm_classifier.fit(x_train, y_train)

#predicting
svm_y_pred= svm_classifier.predict(x_valid)
svm_y_test= svm_classifier.predict(x_test)
svm_y_test= pd.DataFrame(svm_y_test)

#K-fold cross validation
from sklearn.model_selection import cross_val_score
svm_accuracies= cross_val_score(estimator= svm_classifier, 
                                X= x_train, 
                                y= y_train,
                                cv=10)
svm_acc= svm_accuracies.mean()

#preparing Kaggle submission file
svm_submission= pd.concat([pass_id, svm_y_test], axis=1)
svm_submission.to_csv("svm_submission.csv")

#fitting Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_classifier= GaussianNB()
nb_classifier.fit(x_train, y_train)


#predicting
nb_y_pred= nb_classifier.predict(x_valid)
nb_y_test= nb_classifier.predict(x_test)
nb_y_test= pd.DataFrame(nb_y_test)

#K-fold cross validation
from sklearn.model_selection import cross_val_score
nb_accuracies= cross_val_score(estimator= nb_classifier, 
                                X= x_train, 
                                y= y_train,
                                cv=10)
nb_acc= nb_accuracies.mean()

#preparing Kaggle submission file
nb_submission= pd.concat([pass_id, nb_y_test], axis=1)
nb_submission.to_csv("nb_submission.csv")

#fitting Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc_classifier= DecisionTreeClassifier(criterion="entropy", random_state=0)
dtc_classifier.fit(x_train, y_train)

#predicting
dtc_y_pred= dtc_classifier.predict(x_valid)
dtc_y_test= dtc_classifier.predict(x_test)
dtc_y_test= pd.DataFrame(dtc_y_test)

#K-fold cross validation
from sklearn.model_selection import cross_val_score
dtc_accuracies= cross_val_score(estimator= dtc_classifier, 
                                X= x_train, 
                                y= y_train,
                                cv=10)
dtc_acc= dtc_accuracies.mean()

#preparing Kaggle submission file
dtc_submission= pd.concat([pass_id, dtc_y_test], axis=1)
dtc_submission.to_csv("dtc_submission.csv")


#fitting RF Classification
from sklearn.ensemble import RandomForestClassifier
rfc_classifier= RandomForestClassifier(n_estimators=100, max_depth=5, criterion="entropy", random_state=0)
rfc_classifier.fit(x_train, y_train)

#predicting
rfc_y_pred= rfc_classifier.predict(x_valid)
rfc_y_test= rfc_classifier.predict(x_test)
rfc_y_test= pd.DataFrame(rfc_y_test)

#K-fold cross validation
from sklearn.model_selection import cross_val_score
rfc_accuracies= cross_val_score(estimator= rfc_classifier, 
                                X= x_train, 
                                y= y_train,
                                cv=10)
rfc_acc= dtc_accuracies.mean()

#preparing Kaggle submission file
rfc_submission= pd.concat([pass_id, rfc_y_test], axis=1)
rfc_submission.to_csv("rfc_submission.csv")

#fitting xgboost model
from xgboost import XGBClassifier
xgb_classifier= XGBClassifier()
xgb_classifier.fit(x_train, y_train)

#predicting
xgb_y_pred= xgb_classifier.predict(x_valid)
xgb_y_test= xgb_classifier.predict(x_test)
xgb_y_test= pd.DataFrame(xgb_y_test)

#K-fold cross validation
from sklearn.model_selection import cross_val_score
xgb_accuracies= cross_val_score(estimator= xgb_classifier, 
                                X= x_train, 
                                y= y_train,
                                cv=10)
xgb_acc= xgb_accuracies.mean()

#preparing Kaggle submission file
xgb_submission= pd.concat([pass_id, xgb_y_test], axis=1)
xgb_submission.to_csv("xgb_submission.csv")

#applying Grid Search
from sklearn.model_selection import GridSearchCV
parameters= [{"C": [1,10,100,1000], "kernel":["linear"]},
             {"C": [1,10,100,1000], "kernel":["rbf"], "gamma": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}
            ]
grid_search= GridSearchCV(estimator= svm_classifier,
                          param_grid= parameters,
                          scoring= "accuracy",
                          cv=10,
                          n_jobs= -1)
grid_search= grid_search.fit(x_train, y_train)
best_accuracy= grid_search.best_score_
best_parameters= grid_search.best_params_

        










