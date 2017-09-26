# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.linear_model import LogisticRegression
#from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



''' # # # # # # # # # PREPARATION # # # # # # # # # # '''

# open file
train_df = pd.read_csv("train.csv")
# get labels
train_labels = train_df['Activity']
# remove unecessary labels
del train_df['subject']
del train_df['Activity']




''' # # # # # # # # # # # FUNCTION FOR TESTING TEST SET # # # # # # # # # # '''

# load test set
test_df = pd.read_csv("test.csv")
# get labels
test_labels = test_df['Activity']
# remove unecesary features
del test_df['subject']
del test_df['Activity']
    
# function for testing set
def test(models, feature_names=[]):
    test_copy = test_df.copy()
    # subset if needed
    if len(feature_names) > 0:
        test_copy = test_copy[feature_names]
    # predict
    for model in models:
        predictions = model.predict(test_copy)
        print confusion_matrix(test_labels,predictions)
        print accuracy_score(test_labels,predictions)
        
# BASELINE (RANDOM FOREST)
baseline_rf = RandomForestClassifier(n_estimators=150, max_depth=15, min_samples_split=10, random_state=42)
baseline_rf.fit(train_df, train_labels)

# test it
test(baseline_rf)
# current acccuracy: 0.926705123855




''' # # # # # # # # # DEFINE CLASSIFICATION MODELS # # # # # # # # # # '''

# LogReg
def lr_(X_train, X_test, y_train, y_test):
    clf_lr = LogisticRegression(random_state=42)
    #clf_svm = GridSearchCV(svm.SVC(),{'C':[100.],'kernel':['linear'],'degree':[1],'decision_function_shape':['ovo']}, cv=10)
    clf_lr.fit(X_train, y_train)
    predict_lr = clf_lr.predict(X_test)
    
    print confusion_matrix(y_test,predict_lr)
    #print precision_score(y_test,predict_svm,average=None)
    #print recall_score(y_test,predict_svm,average=None)
    print accuracy_score(y_test,predict_lr)
    return clf_lr

# SVM
def svm_(X_train, X_test, y_train, y_test):
    clf_svm = svm.SVC(decision_function_shape='ovo', C=100.0, degree=1, random_state=42)
    #clf_svm = GridSearchCV(svm.SVC(),{'C':[100.],'kernel':['linear'],'degree':[1],'decision_function_shape':['ovo']}, cv=10)
    clf_svm.fit(X_train, y_train)
    predict_svm = clf_svm.predict(X_test)
    
    print confusion_matrix(y_test,predict_svm)
    #print precision_score(y_test,predict_svm,average=None)
    #print recall_score(y_test,predict_svm,average=None)
    print accuracy_score(y_test,predict_svm)
    return clf_svm

# decision tree
def dt_(X_train, X_test, y_train, y_test):
    clf_tree = tree.DecisionTreeClassifier(criterion='entropy', random_state=5, max_depth=25)
    #clf_tree = GridSearchCV(tree.DecisionTreeClassifier(),{'random_state':[42],'max_depth':[25],'criterion':['entropy']}, cv=10)
    clf_tree.fit(X_train, y_train)
    predict_tree = clf_tree.predict(X_test)
    
    print confusion_matrix(y_test,predict_tree)
    print accuracy_score(y_test,predict_tree)
    return clf_tree

# random forest
def rf_(X_train, X_test, y_train, y_test):
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=15, random_state=42)
    #clf_rf = GridSearchCV(RandomForestClassifier(),{'n_estimators':[100],'max_depth':[15],'criterion':['entropy'], 'min_samples_split':[15]},cv=10)    
    clf_rf.fit(X_train, y_train)
    predict_rf = clf_rf.predict(X_test)
    
    print confusion_matrix(y_test,predict_rf)
    print accuracy_score(y_test,predict_rf)
    return clf_rf

'''
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
'''




''' # # # # # # # # # FEATURE SELECTION # # # # # # # # # # '''

# # #
# # # baseline - try the first n (50) features
# remove unnecessary features
train_df2 = train_df.copy()

# SPLIT THE TRAIN DATA INTO TRAIN AND VALIDATION
X_train, X_test, y_train, y_test = train_test_split(train_df2, train_labels, test_size=0.3, random_state=42)

# choose first 50
X_train = X_train.ix[:,:90]
X_test  = X_test.ix[:,:90]
# EVALUATE
base_svm = svm_(X_train, X_test, y_train, y_test)
# 0.951495920218
base_dt  = dt_(X_train, X_test, y_train, y_test)
# 0.929737080689
base_rf  = rf_(X_train, X_test, y_train, y_test)
# 0.955575702629
'''
nn_(X_train, X_test, y_train, y_test)
# 0.939256572983
'''
############## TESTING #############
features = X_train.columns
test([base_svm, base_dt, base_rf], features)



# # #
# # # remove most highly correlated features
train_df4 = train_df.copy()
corr = train_df4.corr()

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
cor_svm = svm_(X_train, X_test, y_train, y_test)
# 0.914777878513
cor_dt  = dt_(X_train, X_test, y_train, y_test)
# 0.810516772439
cor_rf  = rf_(X_train, X_test, y_train, y_test)
# 0.915684496827
'''
nn_(X_train, X_test, y_train, y_test)
# 0.914324569356
'''
############## TESTING #############
features = train_df4.columns
test([cor_svm, cor_dt, cor_rf], features)
# 0.818120122158 (0.55 and -0.55)

# plot the two least correlated features
walking = train_df.loc[train_df['Activity'] == "WALKING"]
sitting = train_df.loc[train_df['Activity'] == "SITTING"]
laying = train_df.loc[train_df['Activity'] == "LAYING"]
walking_up = train_df.loc[train_df['Activity'] == "WALKING_UPSTAIRS"]
walking_down = train_df.loc[train_df['Activity'] == "WALKING_DOWNSTAIRS"]
standing = train_df.loc[train_df['Activity'] == "STANDING"]
fig= plt.figure()
#plt.scatter(train_df['tBodyGyroMag-arCoeff()4'], train_df['tBodyGyro-min()-Z'],c=colors)
plt.scatter(walking['tBodyGyroMag-arCoeff()4'], walking['tBodyGyro-min()-Z'], color="red", label="WALKING")
plt.scatter(sitting['tBodyGyroMag-arCoeff()4'], sitting['tBodyGyro-min()-Z'], color="blue", label="SITTING")
plt.scatter(laying['tBodyGyroMag-arCoeff()4'], laying['tBodyGyro-min()-Z'], color="green", label="LAYING")
plt.scatter(standing['tBodyGyroMag-arCoeff()4'], standing['tBodyGyro-min()-Z'], color="purple", label="STANDING")
plt.scatter(walking_up['tBodyGyroMag-arCoeff()4'], walking_up['tBodyGyro-min()-Z'], color="orange", label="WALKING UPSTAIRS")
plt.scatter(walking_down['tBodyGyroMag-arCoeff()4'], walking_down['tBodyGyro-min()-Z'], color="black", label="WALKING DOWNSTAIRS")

plt.xlabel('tBodyGyroMag-arCoeff()4')
plt.ylabel('tBodyGyro-min()-Z')
plt.title("Plot of the two least correlated features by activity")
plt.legend(loc='lower left')
fig.savefig("correlation.pdf")
plt.close()






# # # 
# # # take the means grouped by Activity - take the N ones with the largest variance
train_df3 = train_df.copy()
train_df3['Activity'] = train_labels
train_df3 = train_df3.groupby(['Activity']).mean()
var = train_df3.var(axis=0)
df_var = pd.DataFrame({'features':var.index, 'value':var.values})
top_n_var = df_var.nlargest(50,'value')     # 50 largest variances, includes ties
top_n_features = list(top_n_var['features'])

new_train = train_df2[top_n_features]
X_train, X_test, y_train, y_test = train_test_split(new_train, train_labels, test_size=0.3, random_state=42)
# EVALUATE
means_svm = svm_(X_train, X_test, y_train, y_test)
# 0.935630099728
means_dt  = dt_(X_train, X_test, y_train, y_test)
# 0.896645512239
means_rf  = rf_(X_train, X_test, y_train, y_test)
# 0.924297370807
'''
nn_(X_train, X_test, y_train, y_test)
# 0.900725294651
'''
############## TESTING #############
features = new_train.columns
test([means_svm, means_dt, means_rf], features)


# plot top two features by activity
df = df_var.drop(df_var.index[561]) # drops subject id row
topIndex = df['value'].argmax()
top1feature = df.loc[topIndex,:]['features']
df = df.drop(df.index[topIndex])



# # #
# # # remove features with low variance
from sklearn.feature_selection import VarianceThreshold
train_df5 = train_df.copy()
sel = VarianceThreshold(threshold=(0.25))    # rm all features less than this variance
sel.fit(train_df5)
#a = [not i for i in sel.get_support()]      # flip boolean list
a = sel.get_support()
train_df5 = train_df5.loc[:, a]
# train_df5 = sel.fit_transform(train_df5)
X_train, X_test, y_train, y_test = train_test_split(train_df5, train_labels, test_size=0.3, random_state=42)
var_svm = svm_(X_train, X_test, y_train, y_test)
# 0.94968268359
var_dt  = dt_(X_train, X_test, y_train, y_test)
# 0.898458748867
var_rf  = rf_(X_train, X_test, y_train, y_test)
# 0.93970988214
'''
nn_(X_train, X_test, y_train, y_test)
# 0.932003626473
'''
############## TESTING #############
features = train_df5.columns
test([var_svm, var_dt, var_rf], features)



# # #
# # # try using the top n features with highest variance that reaches over 90% accuracy
train_df6 = train_df.copy()
a = train_df6.var(axis=0)
a = a.sort_values(ascending=False)
train_df6 = train_df6[a[:25].index]
X_train, X_test, y_train, y_test = train_test_split(train_df6, train_labels, test_size=0.3, random_state=42)
hvar_svm = svm_(X_train, X_test, y_train, y_test)
# 0.944242973708
hvar_dt  = dt_(X_train, X_test, y_train, y_test)
# 0.901631912965
hvar_rf  = rf_(X_train, X_test, y_train, y_test)
# 0.936083408885
'''
nn_(X_train, X_test, y_train, y_test)
# 0.934270172257
'''
############## TESTING #############
features = train_df6.columns
test([hvar_svm, hvar_dt, hvar_rf], features)



# # #
# # # backwards elimination (take off only ONE feature per iteration)
# NOTE: THIS TAKES AWHILE, all features derived from logistic regression scores #
# the following was acquired after running this code, just so you don't have to rerun it #

# for safekeeping, top 5 features, indices = [40, 41, 102, 233, 268] 
'''[tGravityAcc-mean()-X', u'tGravityAcc-mean()-Y',
       'tBodyAccJerk-entropy()-X', 'tBodyAccJerkMag-iqr()','fBodyAcc-std()-X'] '''
# for safekeeping, top 7 features, indices = [40, 41, 56, 102, 233, 268, 504] 
'''['tGravityAcc-mean()-X', 'tGravityAcc-mean()-Y', 'tGravityAcc-energy()-X', 'tBodyAccJerk-entropy()-X',
       'tBodyAccJerkMag-iqr()', 'fBodyAcc-std()-X', 'fBodyAccMag-mad()']'''
# for safekeeping, top 10 features, indices = [40, 41, 50, 56, 102, 182, 233, 268, 463, 504] 
''' ['tGravityAcc-mean()-X', 'tGravityAcc-mean()-Y', 'tGravityAcc-max()-Y', 'tGravityAcc-energy()-X',
       'tBodyAccJerk-entropy()-X', 'tBodyGyroJerk-entropy()-X', 'tBodyAccJerkMag-iqr()', 'fBodyAcc-std()-X',
       'fBodyGyro-bandsEnergy()-25,32', 'fBodyAccMag-mad()']'''

from sklearn.feature_selection import RFE
train_df7 = train_df.copy()
# commented below is what you would run to get the top N features
'''
X_train, X_test, y_train, y_test = train_test_split(train_df7, train_labels, test_size=0.3, random_state=42)
model = LogisticRegression()
rfe = RFE(model, 5)
rfe = rfe.fit(X_train, y_train)                     # takes a long time
support = rfe.support_
ranking = rfe.ranking_
indices = [i for i, x in enumerate(support) if x]   # indices of features
'''
indices = [40, 41, 102, 233, 268]                           # top 5
indices = [40, 41, 56, 102, 233, 268, 504]                  # top 7
indices = [40, 41, 50, 56, 102, 182, 233, 268, 463, 504]    # top 10
# try new classification
train_df7 = train_df7[indices]
X_train, X_test, y_train, y_test = train_test_split(train_df7, train_labels, test_size=0.3, random_state=42)
be_lr = lr_(X_train, X_test, y_train, y_test)
# 5 - 0.894378966455, 7 - 0.899818676337, 10 - 0.92248413418
be_svm = svm_(X_train, X_test, y_train, y_test)
# 5 - 0.907071622847, 7 - 0.912058023572, 10 - 0.936990027199
be_dt  = dt_(X_train, X_test, y_train, y_test)
# 5 - 0.926110607434, 7 - 0.927470534905, 10 - 0.929283771532
be_rf  = rf_(X_train, X_test, y_train, y_test)
# 5 - 0.937896645512, 7 - 0.944696282865, 10 - 0.954215775159
############## TESTING #############
features = train_df7.columns
test([be_lr, be_svm, be_dt, be_rf], features)
# 5 - 0.894378966455, 0.907071622847, 0.926110607434, 0.937896645512
# 7 - 0.869019341703, 0.869019341703, 0.803529012555, 0.844927044452
# 10 - 0.873769935528, 0.912058023572, 0.927470534905, 0.944696282865



# # #
# # # recursive feature elimination - feature importance from random forest
# NOTE: this also takes some time, here is the output to save some time
# for safekeeping:, indices2 = [ 40,  49, 558,  52, 559,  50,  53,  41,  56, 181]
''' indices2 = ['tGravityAcc-mean()-X', 'tGravityAcc-energy()-X', 'fBodyAccJerk-mean()-X', 'tGravityAcc-min()-X',
       'tBodyAccJerk-std()-X', 'angle(X,gravityMean)', 'tGravityAcc-mean()-Y', 'fBodyAccJerk-max()-X',
       'tBodyGyroJerk-iqr()-Z', 'tGravityAcc-max()-X'] '''
train_df8 = train_df.copy()
# commented below is what you would run to get the top N features
'''
X_train, X_test, y_train, y_test = train_test_split(train_df8, train_labels, test_size=0.3, random_state=42)
importance_rf = RandomForestClassifier(criterion='entropy',n_estimators=200, max_depth=15, min_samples_split=15, random_state=42)
importance_rf.fit(X_train, y_train)
importances = importance_rf.feature_importances_    # not ordered
indices2 = np.argsort(importances)[::-1]            # ordered indices, descending
'''
features = indices2[:30]
train_df8 = train_df8[features]
X_train, X_test, y_train, y_test = train_test_split(train_df8, train_labels, test_size=0.3, random_state=42)
rfe_svm = svm_(X_train, X_test, y_train, y_test)
# 0.885312783318
rfe_dt  = dt_(X_train, X_test, y_train, y_test)
# 0.915684496827
rfe_rf  = rf_(X_train, X_test, y_train, y_test)
# 0.932910244787
############## TESTING #############
features = train_df8.columns
test([rfe_svm, rfe_dt, rfe_rf], features)
# model (# features) - accuracy
# svm (10) - 0.756362402443 : (20) - 0.830675262979
# dt  (10) - 0.709535120461 : (20) - 0.762809636919
# rf  (10) - 0.72989480828  : (20) - 0.765863590092



# # # 
# # # recursive feature elimination with cross validation
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

train_df9 = train_df.copy()
svc = SVC(kernel='linear')
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv.fit(X_train, y_train)

support2 = rfecv.support_
indices3 = [i for i, x in enumerate(support2) if x]
train_df9 = train_df9[indices3]
X_train, X_test, y_train, y_test = train_test_split(train_df9, train_labels, test_size=0.3, random_state=42)
print("Optimal number of features : %d" % rfecv.n_features_)

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

predicts = rfecv.predict(X_test)
print confusion_matrix(y_test,predicts)
print accuracy_score(y_test,predicts)

features = train_df9.columns
test([rfecv],features)


# # #
# # # rfe using svm
from sklearn.svm import SVC
train_df10 = train_df.copy()
X_train, X_test, y_train, y_test = train_test_split(train_df10, train_labels, test_size=0.3, random_state=42)
estimator_lr = LogisticRegression()
estimator_svm = SVC(kernel="linear", decision_function_shape='ovo')
estimator_dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=25)
estimator_rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=15, random_state=42)

selector_lr = RFE(estimator_lr, 5, step=10.0)
selector_lr = selector_lr.fit(X_train, y_train)
#X_test = X_test.loc[:,selector_lr.get_support()]
predicts_lr = selector_lr.predict(X_test)
print confusion_matrix(y_test,predicts_lr)
print accuracy_score(y_test,predicts_lr)
features_lr = list(train_df.loc[:,selector_lr.get_support()].columns)

selector_svm = RFE(estimator_svm, 5, step=10.0)
selector_svm = selector_svm.fit(X_train, y_train)
predicts_svm = selector_svm.predict(X_test)
print confusion_matrix(y_test,predicts_svm)
print accuracy_score(y_test,predicts_svm)
features_svm = list(train_df.loc[:,selector_svm.get_support()].columns)
                           
selector_dt = RFE(estimator_dt, 5, step=10.0)
selector_dt = selector_dt.fit(X_train, y_train)
predicts_dt = selector_dt.predict(X_test)
print confusion_matrix(y_test,predicts_dt)
print accuracy_score(y_test,predicts_dt)
features_dt = list(train_df.loc[:,selector_dt.get_support()].columns)
                           
selector_rf = RFE(estimator_rf, 5, step=10.0)
selector_rf = selector_rf.fit(X_train, y_train)
predicts_rf = selector_rf.predict(X_test)
print confusion_matrix(y_test,predicts_rf)
print accuracy_score(y_test,predicts_rf)
features_rf = list(train_df.loc[:,selector_rf.get_support()].columns)
                           
test_predicts_lr = selector_lr.predict(test_df)
print accuracy_score(test_labels,test_predicts_lr)
test_predicts_svm = selector_svm.predict(test_df)
print accuracy_score(test_labels,test_predicts_svm)
test_predicts_dt = selector_dt.predict(test_df)
print accuracy_score(test_labels,test_predicts_dt)
test_predicts_rf = selector_rf.predict(test_df)
print accuracy_score(test_labels,test_predicts_rf)





# # #
# TESTING ONLY
# # # best one for 80% using 3 features
h = [40, 41, 181] 
train_df9 = train_df.copy()
train_df9 = train_df9[h]
# ['tGravityAcc-mean()-X', 'fBodyAccMag-mad()', 'tBodyGyroJerk-iqr()-Z']
X_train, X_test, y_train, y_test = train_test_split(train_df9, train_labels, test_size=0.3, random_state=42)
clf = GridSearchCV(svm.SVC(),{'C':[100],'kernel':['linear'],'degree':[1],'decision_function_shape':['ovo']}, cv=10)
clf.fit(X_train,y_train)
predict = clf.predict(X_test)
print accuracy_score(y_test,predict)

from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(train_df9):
    X_train, X_test = train_df9.iloc[train_index], train_df9.iloc[test_index]
    y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]
    clf = svm.SVC(C=100, decision_function_shape='ovo', random_state=42)
    clf.fit(X_train,y_train)
    predict = clf.predict(X_test)
    print accuracy_score(y_test,predict)

svm_ = svm_(X_train, X_test, y_train, y_test)
# 0.885312783318
dt_  = dt_(X_train, X_test, y_train, y_test)
# 0.915684496827
rf_  = rf_(X_train, X_test, y_train, y_test)
# 0.932910244787
############## TESTING #############
features = train_df9.columns
test([svm_, dt_, rf_], features)


''' # # # # # # # # # PCA (NOT USED) # # # # # # # # # # '''

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
'''
nn_(X_t_train, X_t_test, pca_train_labels, pca_test_labels)
# 0.932003626473
'''



''' # # # # # # # # # # # # # OUTLIER CHECK # # # # # # # # # # # # # # # '''

# for each column, check if there are >100 values past 3 standard deviations
check2 = train_df.copy()
check2['Activity'] = train_labels
for j in check2.columns:
    check = check2.loc[check2['Activity'] == j]
    del check['Activity']
    v = len(check.index)
    for i in check.columns:
        a = check[np.abs(check[i]-check[i].mean())<=(3*check[i].std())]
        l = len(a.index)
        if (v - l) > (v*.10):
            print i
            print v-l


