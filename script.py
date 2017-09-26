# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


''' # # # # # # # # # PREPARATION # # # # # # # # # # '''

# open file
train_df = pd.read_csv("/Users/rickysu/Desktop/DSC383W/mini_project/train.csv")

# if you want to randomize the data before splitting train set to validation
# train_df = train_df.sample(frac=1, random_state=42)

# get labels
train_labels = train_df['Activity']

# remove unecessary labels
del train_df['subject']


''' # # # # # # # # # DEFINE CLASSIFICATION MODELS # # # # # # # # # # '''

# SVM
def svm_(X_train, X_test, y_train, y_test):
    clf_svm = svm.SVC(decision_function_shape='ovo', C=100.0, degree=1, random_state=42)
    #clf = GridSearchCV(svm.SVC(),{'C':range(10000,50001,100000),'kernel':['linear'],'degree':[1],'decision_function_shape':['ovo']})
    clf_svm.fit(X_train, y_train)
    predict_svm = clf_svm.predict(X_test)
    
    print confusion_matrix(y_test,predict_svm)
    #print precision_score(y_test,predict_svm,average=None)
    #print recall_score(y_test,predict_svm,average=None)
    print accuracy_score(y_test,predict_svm)
    # current accuracy: 0.951495920218 (with first 50 features)

# decision tree
def dt_(X_train, X_test, y_train, y_test):
    clf_tree = tree.DecisionTreeClassifier(criterion='entropy', random_state=5, max_depth=25)
    #clf_tree = GridSearchCV(tree.DecisionTreeClassifier(),{'random_state':range(1,6,1),'max_depth':range(10,16,1),'criterion':['gini','entropy']})
    clf_tree.fit(X_train, y_train)
    predict_tree = clf_tree.predict(X_test)
    
    print confusion_matrix(y_test,predict_tree)
    print accuracy_score(y_test,predict_tree)
    # current accuracy: 0.929737080689 (with first 50 features)

# random forest
def rf_(X_train, X_test, y_train, y_test):
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=15, random_state=42)
    #clf_rf = GridSearchCV(RandomForestClassifier(),{'n_estimators':range(10,101,10),'max_depth':range(10,16,1),'criterion':['gini','entropy'], 'min_samples_split':range(10,20,3)})
    clf_rf.fit(X_train, y_train)
    predict_rf = clf_rf.predict(X_test)
    
    print confusion_matrix(y_test,predict_rf)
    print accuracy_score(y_test,predict_rf)
    # current accuracy: 0.955575702629 (with first 50 features)

# Neural Network
def nn_(X_train, X_test, y_train, y_test):
    clf_deep = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(100,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=42,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=True)
    clf_deep.fit(X_train, y_train)
    predict_deep = clf_deep.predict(X_test)
    
    print confusion_matrix(y_test,predict_deep)
    print accuracy_score(y_test,predict_deep)
    # current accuracy: 0.939256572983 (with first 50 features)


''' # # # # # # # # # FEATURE SELECTION # # # # # # # # # # '''

# # # try taking the means grouped by Activity - take the ones with the largest variance
train_df2 = train_df.copy()
test = train_df2.groupby(['Activity']).mean()
del train_df2['Activity']
var = test.var(axis=0)
df_var = pd.DataFrame({'features':var.index, 'value':var.values})
top_n_var = df_var.nlargest(50,'value')     # 50 largest variances, includes ties
top_n_features = list(top_n_var['features'])

new_train = train_df2[top_n_features]
X_train, X_test, y_train, y_test = train_test_split(new_train, train_labels, test_size=0.3, random_state=42)
# EVALUATE
svm_(X_train, X_test, y_train, y_test)
# 0.935630099728
dt_(X_train, X_test, y_train, y_test)
# 0.896645512239
rf_(X_train, X_test, y_train, y_test)
# 0.924297370807
nn_(X_train, X_test, y_train, y_test)
# 0.900725294651


# # # baseline - try the first n (50) features
# remove unnecessary features
train_df3 = train_df.copy()
del train_df3['Activity']

# SPLIT THE TRAIN DATA INTO TRAIN AND VALIDATION
X_train, X_test, y_train, y_test = train_test_split(train_df, train_labels, test_size=0.3, random_state=42)

# choose first 50
X_train = X_train.ix[:,:50]
X_test  = X_test.ix[:,:50]
# EVALUATE
svm_(X_train, X_test, y_train, y_test)
# 0.951495920218
dt_(X_train, X_test, y_train, y_test)
# 0.929737080689
rf_(X_train, X_test, y_train, y_test)
# 0.955575702629
nn_(X_train, X_test, y_train, y_test)
# 0.939256572983


# # # remove most highly correlated features
train_df4 = train_df.copy()
del train_df4['Activity']
corr = train_df4.corr()
#most_correlated_cols = {col: corr[col].nlargest(5)[1:].index for col in corr}
#df2 = pd.DataFrame({col: train_df4.loc[:, most_correlated_cols[col]] for col in train_df4})[train_df4.columns]

indices = np.where(corr > 0.7)
indices = [(corr.index[x], corr.columns[y]) for x, y in zip(*indices)
                                        if x != y and x < y]
indices2 = np.where(corr < -0.6)
indices2 = [(corr.index[x], corr.columns[y]) for x, y in zip(*indices2)
                                        if x != y and x < y]

f_i = []
s_i = []
for i in indices:
    f_i.append(i[0])
    s_i.append(i[1])

f_i2 = []
s_i2 = []
for i in indices2:
    f_i2.append(i[0])
    s_i2.append(i[1])

non_features = list(set(f_i + f_i2 + s_i + s_i2))

train_df4 = train_df4.drop(non_features,axis=1)
X_train, X_test, y_train, y_test = train_test_split(train_df4, train_labels, test_size=0.3, random_state=42)
# EVALUATE
svm_(X_train, X_test, y_train, y_test)
# 0.914777878513
dt_(X_train, X_test, y_train, y_test)
# 0.810516772439
rf_(X_train, X_test, y_train, y_test)
# 0.915684496827
nn_(X_train, X_test, y_train, y_test)
# 0.914324569356


# # # remove features with low variance
from sklearn.feature_selection import VarianceThreshold
train_df5 = train_df.copy()
del train_df5['Activity']
sel = VarianceThreshold(threshold=(.65 * (1 - .65)))
sel.fit(train_df5)
train_df5 = train_df5.loc[:, sel.get_support()]
# train_df5 = sel.fit_transform(train_df5)
X_train, X_test, y_train, y_test = train_test_split(train_df5, train_labels, test_size=0.3, random_state=42)
svm_(X_train, X_test, y_train, y_test)
# 0.94968268359
dt_(X_train, X_test, y_train, y_test)
# 0.898458748867
rf_(X_train, X_test, y_train, y_test)
# 0.93970988214
nn_(X_train, X_test, y_train, y_test)
# 0.932003626473


# # # try using the top n features with highest variance that reaches over 90%
train_df6 = train_df.copy()
del train_df6['Activity']
a = train_df6.var(axis=0)
a = a.sort_values(ascending=False)
train_df6 = train_df6[a[:25].index]
X_train, X_test, y_train, y_test = train_test_split(train_df6, train_labels, test_size=0.3, random_state=42)
svm_(X_train, X_test, y_train, y_test)
# 0.944242973708
dt_(X_train, X_test, y_train, y_test)
# 0.901631912965
rf_(X_train, X_test, y_train, y_test)
# 0.936083408885
nn_(X_train, X_test, y_train, y_test)
# 0.934270172257


# # # backwards elimination
# NOTE: THIS TAKES AWHILE #
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
train_df7 = train_df.copy()
del train_df7['Activity']
X_train, X_test, y_train, y_test = train_test_split(train_df7, train_labels, test_size=0.3, random_state=42)
model = LogisticRegression()
rfe = RFE(model, 7)
rfe = rfe.fit(X_train, y_train)     # takes a long time
support = rfe.support_
ranking = rfe.ranking_
indices = [i for i, x in enumerate(support) if x]
# for safekeeping, indices = [40, 41, 50, 56, 102, 182, 233, 268, 463, 504] - logreg # 0.930190389846
''' ['tGravityAcc-mean()-X', 'tGravityAcc-mean()-Y', 'tGravityAcc-max()-Y', 'tGravityAcc-energy()-X',
       'tBodyAccJerk-entropy()-X', 'tBodyGyroJerk-entropy()-X', 'tBodyAccJerkMag-iqr()', 'fBodyAcc-std()-X',
       'fBodyGyro-bandsEnergy()-25,32', 'fBodyAccMag-mad()']'''
# for safekeeping, indices = [40, 41, 56, 102, 233, 268, 504] - logreg # 0.90661831369
'''['tGravityAcc-mean()-X', 'tGravityAcc-mean()-Y', 'tGravityAcc-energy()-X', 'tBodyAccJerk-entropy()-X',
       'tBodyAccJerkMag-iqr()', 'fBodyAcc-std()-X', 'fBodyAccMag-mad()']'''
# try new classification
train_df7 = train_df7[indices]
X_train, X_test, y_train, y_test = train_test_split(train_df7, train_labels, test_size=0.3, random_state=42)
logreg = LogisticRegression(C=1e5)
logreg.fit(X_train, y_train)
predict_logreg = logreg.predict(X_test)
print confusion_matrix(y_test,predict_logreg)
print accuracy_score(y_test,predict_logreg) # 0.930190389846


# # # feature importance from random forest
train_df8 = train_df.copy()
del train_df8['Activity']
X_train, X_test, y_train, y_test = train_test_split(train_df8, train_labels, test_size=0.3, random_state=42)
importance_rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=15, random_state=42)
importance_rf.fit(X_train, y_train)
importances = importance_rf.feature_importances_    # not ordered
indices2 = np.argsort(importances)[::-1]            # ordered indices, descending
# for safekeeping:, indices2 = [ 40,  49, 558,  52, 559,  50,  53,  41,  56, 181]
''' ['tGravityAcc-mean()-X', 'tGravityAcc-max()-X', 'angle(X,gravityMean)', 'tGravityAcc-min()-X', 
     'angle(Y,gravityMean)', 'tGravityAcc-max()-Y', 'tGravityAcc-min()-Y', 'tGravityAcc-mean()-Y', 
     'tGravityAcc-energy()-X', 'tBodyGyroJerk-iqr()-Z'] '''
features = indices2[:10]
train_df8 = train_df8[features]
X_train, X_test, y_train, y_test = train_test_split(train_df8, train_labels, test_size=0.3, random_state=42)
rf_(X_train, X_test, y_train, y_test) # 0.93245693563

   
# TESTING ONLY
h = [40, 41, 181] 
train_df9 = train_df.copy()
train_df9 = train_df9[h]
# ['tGravityAcc-mean()-X', 'fBodyAccMag-mad()', 'tBodyGyroJerk-iqr()-Z']
X_train, X_test, y_train, y_test = train_test_split(train_df9, train_labels, test_size=0.3, random_state=42)
svm_(X_train, X_test, y_train, y_test)
# 0.800543970988



''' # # # # # # # # # EXPLORATION # # # # # # # # # # '''

train_df_pca = train_df.copy()
del train_df_pca['Activity']
X_train, X_test, y_train, y_test = train_test_split(train_df_pca, train_labels, test_size=0.3, random_state=42)

# try PCA
pca_train_labels = y_train.astype('category').cat.codes     # convert to int
pca_test_labels = y_test.astype('category').cat.codes       
pca = PCA(n_components=15)  # choose num components
pca.fit(X_train)
X_t_train = pca.transform(X_train)
X_t_test  = pca.transform(X_test)

svm_(X_t_train, X_t_test, pca_train_labels, pca_test_labels)
# 0.946056210335
dt_(X_t_train, X_t_test, pca_train_labels, pca_test_labels)
# 0.859927470535
rf_(X_t_train, X_t_test, pca_train_labels, pca_test_labels)
# 0.920217588395
nn_(X_t_train, X_t_test, pca_train_labels, pca_test_labels)
# 0.932003626473



''' # # # # # # # # # # # # # TEST SET ONLY # # # # # # # # # # # # # # # '''
         
# test trial
test_df = pd.read_csv("/Users/rickysu/Desktop/DSC383W/mini_project/test.csv")

# get labels
test_labels = test_df['Activity']

# remove unnecessary features
del train_df['Activity']
del test_df['subject']
del test_df['Activity']

# BASELINE (RANDOM FOREST)
baseline_rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=10)
baseline_rf.fit(train_df, train_labels)
baseline_predict = baseline_rf.predict(test_df)
print confusion_matrix(test_labels,baseline_predict)
print accuracy_score(test_labels,baseline_predict)
# current acccuracy: 0.927383780115