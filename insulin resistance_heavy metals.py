#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from eli5.sklearn import PermutationImportance
from pdpbox import info_plots,pdp,get_dataset
import shap
import eli5
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('insulin resistance.csv')
X = df.drop('IR', axis=1)
y = df['IR']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


# RF
rf = RandomForestClassifier(n_estimators=243, min_samples_leaf=3, max_depth=54, max_features='sqrt',min_samples_split=2)
result_rf = rf.fit(X_train, y_train)
print(result_rf.score(X_train, y_train))
print(result_rf.score(X_test, y_test))


# In[ ]:


# XGB
xgb = XGBClassifier(n_estimators=150, max_depth=5, min_child_weight=1,gamma=0, subsample=0.8,
                        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, learning_rate=0.1)
result_xgb = xgb.fit(X_train, y_train)
print(result_xgb.score(X_train, y_train))
print(result_xgb.score(X_test, y_test))


# In[ ]:


# LR
lr = linear_model.LogisticRegression(C=1, fit_intercept=True, tol=1e-4, max_iter=150)
result_lr = lr.fit(X_train, y_train)
print(result_lr.score(X_train, y_train))
print(result_lr.score(X_test, y_test))


# In[ ]:


# GNB
gnb = GaussianNB(priors=None, var_smoothing=1e-10)
result_gnb = gnb.fit(X_train, y_train)
print(result_gnb.score(X_train, y_train))
print(result_gnb.score(X_test, y_test))


# In[ ]:


# RR
rr = linear_model.RidgeClassifier(alpha=0.5, fit_intercept=True, tol=1e-4)
result_rr = rr.fit(X_train, y_train)
print(result_rr.score(X_train, y_train))
print(result_rr.score(X_test, y_test))


# In[ ]:


# SVM
svm = SVC(C=1, kernel='linear', degree=3)
result_svm = svm.fit(X_train, y_train)
print(result_svm.score(X_train, y_train))
print(result_svm.score(X_test, y_test))


# In[ ]:


# MLP
mlp = MLPClassifier(solver='lbfgs', activation='relu', learning_rate_init=0.001, alpha=1e-5,
                    hidden_layer_sizes=(10,50), random_state=2)
result_mlp = mlp.fit(X_train, y_train)
print(result_mlp.score(X_train, y_train))
print(result_mlp.score(X_test, y_test))


# In[ ]:


# DT
dt = DecisionTreeClassifier(criterion="gini", max_depth=6)
result_dt = dt.fit(X_train, y_train)
print(result_dt.score(X_train, y_train))
print(result_dt.score(X_test, y_test))


# In[ ]:


# AB
ab = AdaBoostClassifier(n_estimators=60, learning_rate=1, algorithm='SAMME.R')
result_ab = ab.fit(X_train, y_train)
print(result_ab.score(X_train, y_train))
print(result_ab.score(X_test, y_test))


# In[ ]:


# GBDT
gbdt = GradientBoostingClassifier(n_estimators=31, learning_rate=0.31, max_depth=3, criterion='friedman_mse')
result_gbdt = gbdt.fit(X_train, y_train)
print(result_gbdt.score(X_train, y_train))
print(result_gbdt.score(X_test, y_test))


# In[ ]:


# VC
clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier(criterion="gini", max_depth=6)
clf3 = GaussianNB()
vc = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('gnb', clf3)], voting='soft', flatten_transform='True')
result_vc = vc.fit(X_train, y_train)
print(result_vc.score(X_train, y_train))
print(result_vc.score(X_test, y_test))


# In[ ]:


# KNN
knn = KNeighborsClassifier(n_neighbors=30, algorithm='auto', metric='minkowski', leaf_size=30, weights='uniform')
result_knn = knn.fit(X_train, y_train)
print(result_knn.score(X_train, y_train))
print(result_knn.score(X_test, y_test))


# In[ ]:


# confusion matrix
y_test_pred_rf = rf.predict(X_test)
y_test_pred_proba_rf = rf.predict_proba(X_test)[:,1]
y_train_pred_rf = rf.predict(X_train)
y_train_pred_proba_rf = rf.predict_proba(X_train)[:,1]

y_test_pred_xgb = xgb.predict(X_test)
y_test_pred_proba_xgb = xgb.predict_proba(X_test)[:,1]
y_train_pred_xgb = xgb.predict(X_train)
y_train_pred_proba_xgb = xgb.predict_proba(X_train)[:,1]

y_test_pred_lr = lr.predict(X_test)
y_test_pred_proba_lr = lr.predict_proba(X_test)[:,1]
y_train_pred_lr = lr.predict(X_train)
y_train_pred_proba_lr = lr.predict_proba(X_train)[:,1]

y_test_pred_gnb = gnb.predict(X_test)
y_test_pred_proba_gnb = gnb.predict_proba(X_test)[:,1]
y_train_pred_gnb = gnb.predict(X_train)
y_train_pred_proba_gnb = gnb.predict_proba(X_train)[:,1]

y_test_pred_rr = rr.predict(X_test)
y_test_pred_proba_rr = rr.decision_function(X_test)
y_train_pred_rr = rr.predict(X_train)
y_train_pred_proba_rr = rr.decision_function(X_train)

y_test_pred_svm = svm.predict(X_test)
y_test_pred_proba_svm = svm.decision_function(X_test)
y_train_pred_svm = svm.predict(X_train)
y_train_pred_proba_svm = svm.decision_function(X_train)

y_test_pred_mlp = mlp.predict(X_test)
y_test_pred_proba_mlp = mlp.predict_proba(X_test)[:,1]
y_train_pred_mlp = mlp.predict(X_train)
y_train_pred_proba_mlp = mlp.predict_proba(X_train)[:,1]

y_test_pred_dt = dt.predict(X_test)
y_test_pred_proba_dt = dt.predict_proba(X_test)[:,1]
y_train_pred_dt = dt.predict(X_train)
y_train_pred_proba_dt = dt.predict_proba(X_train)[:,1]

y_test_pred_ab = ab.predict(X_test)
y_test_pred_proba_ab = ab.predict_proba(X_test)[:,1]
y_train_pred_ab = ab.predict(X_train)
y_train_pred_proba_ab = ab.predict_proba(X_train)[:,1]

y_test_pred_gbdt = gbdt.predict(X_test)
y_test_pred_proba_gbdt = gbdt.predict_proba(X_test)[:,1]
y_train_pred_gbdt = gbdt.predict(X_train)
y_train_pred_proba_gbdt = gbdt.predict_proba(X_train)[:,1]

y_test_pred_vc = vc.predict(X_test)
y_test_pred_proba_vc = vc.predict_proba(X_test)[:,1]
y_train_pred_vc = vc.predict(X_train)
y_train_pred_proba_vc = vc.predict_proba(X_train)[:,1]

y_test_pred_knn = knn.predict(X_test)
y_test_pred_proba_knn = knn.predict_proba(X_test)[:,1]
y_train_pred_knn = knn.predict(X_train)
y_train_pred_proba_knn = knn.predict_proba(X_train)[:,1]

confmat_rf = confusion_matrix(y_test, y_test_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=confmat_rf,display_labels=["Healthy","Disease"])
disp.plot(cmap="Reds")
plt.title('Random forest',fontsize=19)
plt.savefig('fig\\confmat\\confmat_rf.svg',dpi= 300, format = "svg")

confmat_xgb = confusion_matrix(y_test, y_test_pred_xgb)
disp = ConfusionMatrixDisplay(confusion_matrix=confmat_xgb,display_labels=["Healthy","Disease"])
disp.plot(cmap="Reds")
plt.title('XGBoost',fontsize=19)
plt.savefig('fig\\confmat\\confmat_xgb.svg',dpi= 300, format = "svg")

confmat_lr = confusion_matrix(y_test, y_test_pred_lr)
disp = ConfusionMatrixDisplay(confusion_matrix=confmat_lr,display_labels=["Healthy","Disease"])
disp.plot(cmap="Reds")
plt.title('Logistic regression',fontsize=19)
plt.savefig('fig\\confmat\\confmat_lr.svg',dpi= 300, format = "svg")

confmat_gnb = confusion_matrix(y_test, y_test_pred_gnb)
disp = ConfusionMatrixDisplay(confusion_matrix=confmat_gnb,display_labels=["Healthy","Disease"])
disp.plot(cmap="Reds")
plt.title('GaussianNB',fontsize=19)
plt.savefig('fig\\confmat\\confmat_gnb.svg',dpi= 300, format = "svg")

confmat_rr = confusion_matrix(y_test, y_test_pred_rr)
disp = ConfusionMatrixDisplay(confusion_matrix=confmat_rr,display_labels=["Healthy","Disease"])
disp.plot(cmap="Reds")
plt.title('Ridge regression',fontsize=19)
plt.savefig('fig\\confmat\\confmat_rr.svg',dpi= 300, format = "svg")

confmat_svm = confusion_matrix(y_test, y_test_pred_svm)
disp = ConfusionMatrixDisplay(confusion_matrix=confmat_svm,display_labels=["Healthy","Disease"])
disp.plot(cmap="Reds")
plt.title('Support vector machine',fontsize=19)
plt.savefig('fig\\confmat\\confmat_svm.svg',dpi= 300, format = "svg")

confmat_mlp = confusion_matrix(y_test, y_test_pred_mlp)
disp = ConfusionMatrixDisplay(confusion_matrix=confmat_mlp,display_labels=["Healthy","Disease"])
disp.plot(cmap="Reds")
plt.title('Multilayer perceptron',fontsize=19)
plt.savefig('fig\\confmat\\confmat_mlp.svg',dpi= 300, format = "svg")

confmat_dt = confusion_matrix(y_test, y_test_pred_dt)
disp = ConfusionMatrixDisplay(confusion_matrix=confmat_dt,display_labels=["Healthy","Disease"])
disp.plot(cmap="Reds")
plt.title('Decision tree',fontsize=19)
plt.savefig('fig\\confmat\\confmat_dt.svg',dpi= 300, format = "svg")

confmat_ab = confusion_matrix(y_test, y_test_pred_ab)
disp = ConfusionMatrixDisplay(confusion_matrix=confmat_ab,display_labels=["Healthy","Disease"])
disp.plot(cmap="Reds")
plt.title('AdaBoost',fontsize=19)
plt.savefig('fig\\confmat\\confmat_ab.svg',dpi= 300, format = "svg")

confmat_gbdt = confusion_matrix(y_test, y_test_pred_gbdt)
disp = ConfusionMatrixDisplay(confusion_matrix=confmat_gbdt,display_labels=["Healthy","Disease"])
disp.plot(cmap="Reds")
plt.title('Gradient Boosting Decision Tree',fontsize=19)
plt.savefig('fig\\confmat\\confmat_gbdt.svg',dpi= 300, format = "svg")

confmat_vc = confusion_matrix(y_test, y_test_pred_vc)
disp = ConfusionMatrixDisplay(confusion_matrix=confmat_vc,display_labels=["Healthy","Disease"])
disp.plot(cmap="Reds")
plt.title('Voting Classifier',fontsize=19)
plt.savefig('fig\\confmat\\confmat_vc.svg',dpi= 300, format = "svg")

confmat_knn = confusion_matrix(y_test, y_test_pred_knn)
disp = ConfusionMatrixDisplay(confusion_matrix=confmat_knn,display_labels=["Healthy","Disease"])
disp.plot(cmap="Reds")
plt.title('K-Nearest Neighbour',fontsize=19)
plt.savefig('fig\\confmat\\confmat_knn.svg',dpi= 300, format = "svg")


# In[ ]:


# AUC
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, y_test_pred_proba_rf)
rft_fpr, rft_tpr, rft_thresholds = roc_curve(y_train, y_train_pred_proba_rf)
rf_auc = auc(rf_fpr, rf_tpr)
rft_auc = auc(rft_fpr, rft_tpr)

xgb_fpr, xgb_tpr, xgb_thresholds = roc_curve(y_test, y_test_pred_proba_xgb)
xgbt_fpr, xgbt_tpr, xgbt_thresholds = roc_curve(y_train, y_train_pred_proba_xgb)
xgb_auc = auc(xgb_fpr, xgb_tpr)
xgbt_auc = auc(xgbt_fpr, xgbt_tpr)

lr_fpr, lr_tpr, lr_thresholds = roc_curve(y_test, y_test_pred_proba_lr)
lrt_fpr, lrt_tpr, lrt_thresholds = roc_curve(y_train, y_train_pred_proba_lr)
lr_auc = auc(lr_fpr, lr_tpr)
lrt_auc = auc(lrt_fpr, lrt_tpr)

gnb_fpr, gnb_tpr, gnb_thresholds = roc_curve(y_test, y_test_pred_proba_gnb)
gnbt_fpr, gnbt_tpr, gnbt_thresholds = roc_curve(y_train, y_train_pred_proba_gnb)
gnb_auc = auc(gnb_fpr, gnb_tpr)
gnbt_auc = auc(gnbt_fpr, gnbt_tpr)

rr_fpr, rr_tpr, rr_thresholds = roc_curve(y_test, y_test_pred_proba_rr)
rrt_fpr, rrt_tpr, rrt_thresholds = roc_curve(y_train, y_train_pred_proba_rr)
rr_auc = auc(rr_fpr, rr_tpr)
rrt_auc = auc(rrt_fpr, rrt_tpr)

svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test, y_test_pred_proba_svm)
svmt_fpr, svmt_tpr, svmt_thresholds = roc_curve(y_train, y_train_pred_proba_svm)
svm_auc = auc(svm_fpr, svm_tpr)
svmt_auc = auc(svmt_fpr, svmt_tpr)

mlp_fpr, mlp_tpr, mlp_thresholds = roc_curve(y_test, y_test_pred_proba_mlp)
mlpt_fpr, mlpt_tpr, mlpt_thresholds = roc_curve(y_train, y_train_pred_proba_mlp)
mlp_auc = auc(mlp_fpr, mlp_tpr)
mlpt_auc = auc(mlpt_fpr, mlpt_tpr)

dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test, y_test_pred_proba_dt)
dtt_fpr, dtt_tpr, dtt_thresholds = roc_curve(y_train, y_train_pred_proba_dt)
dt_auc = auc(dt_fpr, dt_tpr)
dtt_auc = auc(dtt_fpr, dtt_tpr)

ab_fpr, ab_tpr, ab_thresholds = roc_curve(y_test, y_test_pred_proba_ab)
abt_fpr, abt_tpr, abt_thresholds = roc_curve(y_train, y_train_pred_proba_ab)
ab_auc = auc(ab_fpr, ab_tpr)
abt_auc = auc(abt_fpr, abt_tpr)

gbdt_fpr, gbdt_tpr, gbdt_thresholds = roc_curve(y_test, y_test_pred_proba_gbdt)
gbdtt_fpr, gbdtt_tpr, gbdtt_thresholds = roc_curve(y_train, y_train_pred_proba_gbdt)
gbdt_auc = auc(gbdt_fpr, gbdt_tpr)
gbdtt_auc = auc(gbdtt_fpr, gbdtt_tpr)

vc_fpr, vc_tpr, vc_thresholds = roc_curve(y_test, y_test_pred_proba_vc)
vct_fpr, vct_tpr, vct_thresholds = roc_curve(y_train, y_train_pred_proba_vc)
vc_auc = auc(vc_fpr, vc_tpr)
vct_auc = auc(vct_fpr, vct_tpr)

knn_fpr, knn_tpr,knnf_thresholds = roc_curve(y_test, y_test_pred_proba_knn)
knnt_fpr, knnt_tpr, knnt_thresholds = roc_curve(y_train, y_train_pred_proba_knn)
knn_auc = auc(knn_fpr,knn_tpr)
knnt_auc = auc(knnt_fpr, knnt_tpr)


# In[ ]:


# ROC
plt.figure(figsize=(5.8,4.5 ))
plt.plot(rf_fpr, rf_tpr, c = '#F0988C',linewidth=3)
plt.plot(rft_fpr, rft_tpr, c = '#A1A9D0',linewidth=3)
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('Random forest',fontsize=19)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.56,0.15,fontdict=None, s = 'Testing   AUC = %.3f'%rf_auc, c = 'black',backgroundcolor='#F0988C')
plt.text(0.56,0.08,fontdict=None, s = 'Training  AUC = %.3f'%rft_auc, c = 'black',backgroundcolor='#A1A9D0')
plt.grid(True,linewidth=2.5)
bwith = 2.5 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.savefig('fig\\roc\\roc_rf.svg',dpi= 300, format = "svg")

plt.figure(figsize=(5.8,4.5 ))
plt.plot(xgb_fpr, xgb_tpr, c = '#F0988C',linewidth=3)
plt.plot(xgbt_fpr, xgbt_tpr, c = '#A1A9D0',linewidth=3)
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('XGBoost',fontsize=19)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.56,0.15,fontdict=None, s = 'Testing   AUC = %.3f'%xgb_auc, c = 'black',backgroundcolor='#F0988C')
plt.text(0.56,0.08,fontdict=None, s = 'Training  AUC = %.3f'%xgbt_auc, c = 'black',backgroundcolor='#A1A9D0')
plt.grid(True,linewidth=2.5)
bwith = 2.5 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.savefig('fig\\roc\\roc_xgb.svg',dpi= 300, format = "svg")

plt.figure(figsize=(5.8,4.5 ))
plt.plot(lr_fpr, lr_tpr, c = '#F0988C',linewidth=3)
plt.plot(lrt_fpr, lrt_tpr, c = '#A1A9D0',linewidth=3)
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('Logistic regression',fontsize=19)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.56,0.15,fontdict=None, s = 'Testing   AUC = %.3f'%lr_auc, c = 'black',backgroundcolor='#F0988C')
plt.text(0.56,0.08,fontdict=None, s = 'Training  AUC = %.3f'%lrt_auc, c = 'black',backgroundcolor='#A1A9D0')
plt.grid(True,linewidth=2.5)
bwith = 2.5 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.savefig('fig\\roc\\roc_lr.svg',dpi= 300, format = "svg")

plt.figure(figsize=(5.8,4.5 ))
plt.plot(gnb_fpr, gnb_tpr, c = '#F0988C',linewidth=3)
plt.plot(gnbt_fpr, gnbt_tpr, c = '#A1A9D0',linewidth=3)
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('GaussianNB',fontsize=19)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.56,0.15,fontdict=None, s = 'Testing   AUC = %.3f'%gnb_auc, c = 'black',backgroundcolor='#F0988C')
plt.text(0.56,0.08,fontdict=None, s = 'Training  AUC = %.3f'%gnbt_auc, c = 'black',backgroundcolor='#A1A9D0')
plt.grid(True,linewidth=2.5)
bwith = 2.5 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.savefig('fig\\roc\\roc_gnb.svg',dpi= 300, format = "svg")

plt.figure(figsize=(5.8,4.5 ))
plt.plot(rr_fpr, rr_tpr, c = '#F0988C',linewidth=3)
plt.plot(rrt_fpr, rrt_tpr, c = '#A1A9D0',linewidth=3)
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('Ridge regression',fontsize=19)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.56,0.15,fontdict=None, s = 'Testing   AUC = %.3f'%rr_auc, c = 'black',backgroundcolor='#F0988C')
plt.text(0.56,0.08,fontdict=None, s = 'Training  AUC = %.3f'%rrt_auc, c = 'black',backgroundcolor='#A1A9D0')
plt.grid(True,linewidth=2.5)
bwith = 2.5 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.savefig('fig\\roc\\roc_rr.svg',dpi= 300, format = "svg")

plt.figure(figsize=(5.8,4.5 ))
plt.plot(svm_fpr, svm_tpr, c = '#F0988C',linewidth=3)
plt.plot(svmt_fpr, svmt_tpr, c = '#A1A9D0',linewidth=3)
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('Support vector machine',fontsize=19)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.56,0.15,fontdict=None, s = 'Testing   AUC = %.3f'%svm_auc, c = 'black',backgroundcolor='#F0988C')
plt.text(0.56,0.08,fontdict=None, s = 'Training  AUC = %.3f'%svmt_auc, c = 'black',backgroundcolor='#A1A9D0')
plt.grid(True,linewidth=2.5)
bwith = 2.5 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.savefig('fig\\roc\\roc_svm.svg',dpi= 300, format = "svg")

plt.figure(figsize=(5.8,4.5 ))
plt.plot(mlp_fpr, mlp_tpr, c = '#F0988C',linewidth=3)
plt.plot(mlpt_fpr, mlpt_tpr, c = '#A1A9D0',linewidth=3)
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('Multilayer perceptron',fontsize=19)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.56,0.15,fontdict=None, s = 'Testing   AUC = %.3f'%mlp_auc, c = 'black',backgroundcolor='#F0988C')
plt.text(0.56,0.08,fontdict=None, s = 'Training  AUC = %.3f'%mlpt_auc, c = 'black',backgroundcolor='#A1A9D0')
plt.grid(True,linewidth=2.5)
bwith = 2.5 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.savefig('fig\\roc\\roc_mlp.svg',dpi= 300, format = "svg")

plt.figure(figsize=(5.8,4.5 ))
plt.plot(dt_fpr, dt_tpr, c = '#F0988C',linewidth=3)
plt.plot(dtt_fpr, dtt_tpr, c = '#A1A9D0',linewidth=3)
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('Decision tree',fontsize=19)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.56,0.15,fontdict=None, s = 'Testing   AUC = %.3f'%dt_auc, c = 'black',backgroundcolor='#F0988C')
plt.text(0.56,0.08,fontdict=None, s = 'Training  AUC = %.3f'%dtt_auc, c = 'black',backgroundcolor='#A1A9D0')
plt.grid(True,linewidth=2.5)
bwith = 2.5 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.savefig('fig\\roc\\roc_dt.svg',dpi= 300, format = "svg")

plt.figure(figsize=(5.8,4.5 ))
plt.plot(ab_fpr, ab_tpr, c = '#F0988C',linewidth=3)
plt.plot(abt_fpr, abt_tpr, c = '#A1A9D0',linewidth=3)
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('AdaBoost',fontsize=19)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.56,0.15,fontdict=None, s = 'Testing   AUC = %.3f'%ab_auc, c = 'black',backgroundcolor='#F0988C')
plt.text(0.56,0.08,fontdict=None, s = 'Training  AUC = %.3f'%abt_auc, c = 'black',backgroundcolor='#A1A9D0')
plt.grid(True,linewidth=2.5)
bwith = 2.5 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.savefig('fig\\roc\\roc_ab.svg',dpi= 300, format = "svg")

plt.figure(figsize=(5.8,4.5 ))
plt.plot(gbdt_fpr, gbdt_tpr, c = '#F0988C',linewidth=3)
plt.plot(gbdtt_fpr, gbdtt_tpr, c = '#A1A9D0',linewidth=3)
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('Gradient Boosting Decision Tree',fontsize=19)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.56,0.15,fontdict=None, s = 'Testing   AUC = %.3f'%gbdt_auc, c = 'black',backgroundcolor='#F0988C')
plt.text(0.56,0.08,fontdict=None, s = 'Training  AUC = %.3f'%gbdtt_auc, c = 'black',backgroundcolor='#A1A9D0')
plt.grid(True,linewidth=2.5)
bwith = 2.5 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.savefig('fig\\roc\\roc_gbdt.svg',dpi= 300, format = "svg")

plt.figure(figsize=(5.8,4.5 ))
plt.plot(vc_fpr, vc_tpr, c = '#F0988C',linewidth=3)
plt.plot(vct_fpr, vct_tpr, c = '#A1A9D0',linewidth=3)
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('Voting Classifier',fontsize=19)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.56,0.15,fontdict=None, s = 'Testing   AUC = %.3f'%vc_auc, c = 'black',backgroundcolor='#F0988C')
plt.text(0.56,0.08,fontdict=None, s = 'Training  AUC = %.3f'%vct_auc, c = 'black',backgroundcolor='#A1A9D0')
plt.grid(True,linewidth=2.5)
bwith = 2.5 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.savefig('fig\\roc\\roc_vc.svg',dpi= 300, format = "svg")

plt.figure(figsize=(5.8,4.5 ))
plt.plot(knn_fpr, knn_tpr, c = '#F0988C',linewidth=3)
plt.plot(knnt_fpr, knnt_tpr, c = '#A1A9D0',linewidth=3)
plt.plot([0, 1], [0, 1],ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('K-Nearest Neighbour',fontsize=19)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.text(0.56,0.15,fontdict=None, s = 'Testing   AUC = %.3f'%knn_auc, c = 'black',backgroundcolor='#F0988C')
plt.text(0.56,0.08,fontdict=None, s = 'Training  AUC = %.3f'%knnt_auc, c = 'black',backgroundcolor='#A1A9D0')
plt.grid(True,linewidth=2.5)
bwith = 2.5 
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.savefig('fig\\roc\\roc_knn.svg',dpi= 300, format = "svg")


# In[ ]:


# feature importance analysis
eli5.show_weights(rf,feature_names=X_train.columns.to_list(), top=43)


# In[ ]:


# PDP
base_features = df.columns.values.tolist()
base_features.remove('IR')
print(base_features)

# Urine Ba
fig, axes, summary_df = info_plots.actual_plot(model=rf, X=X_train, feature='U-Ba',feature_name='Urine Ba',predict_kwds={})
plt.savefig('fig\\pdp\\Ba2disease.svg', dpi= 300, format = "svg")
pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_train, model_features=base_features,feature='U-Ba')
fig, axes = pdp.pdp_plot(pdp_dist, 'Urine Ba', center=True, plot_lines=False, frac_to_plot=0.8, plot_pts_dist=True)
plt.savefig('fig\\pdp\\Ba2disease3.svg', dpi= 300, format = "svg")

# Blood Pb
fig, axes, summary_df = info_plots.actual_plot(model=rf, X=X_train, feature='B-Pb', feature_name='Blood Pb',predict_kwds={})
plt.savefig('fig\\pdp\\Pb2disease.svg', dpi= 300, format = "svg")
pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_train, model_features=base_features,feature='B-Pb')
fig, axes = pdp.pdp_plot(pdp_dist, 'Blood Pb', center=True, plot_lines=False, frac_to_plot=0.8, plot_pts_dist=True)
plt.savefig('fig\\pdp\\Pb2disease3.svg', dpi= 300, format = "svg")

# Blood Cd
fig, axes, summary_df = info_plots.actual_plot(model=rf, X=X_train, feature='B-Cd', feature_name='Blood Cd',predict_kwds={})
plt.savefig('fig\\pdp\\Cd2disease.svg', dpi= 300, format = "svg")
pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_train, model_features=base_features,feature='B-Cd')
fig, axes = pdp.pdp_plot(pdp_dist, 'Blood Cd', center=True, plot_lines=False, frac_to_plot=0.8, plot_pts_dist=True)
plt.savefig('fig\\pdp\\Cd2disease3.svg', dpi= 300, format = "svg")

# Urine Mo
fig, axes, summary_df = info_plots.actual_plot(model=rf, X=X_train, feature='U-Mo', feature_name='Urine Mo',predict_kwds={})
plt.savefig('fig\\pdp\\Mo2disease.svg', dpi= 300, format = "svg")
pdp_dist = pdp.pdp_isolate(model=rf, dataset=X_train, model_features=base_features,feature='U-Mo')
fig, axes = pdp.pdp_plot(pdp_dist, 'Urine Mo', center=True, plot_lines=False, frac_to_plot=0.8, plot_pts_dist=True)
plt.savefig('fig\\pdp\\Mo2disease3.svg', dpi= 300, format = "svg")


# In[ ]:


# interact
feat_name1 = 'U-Ba'
nick_name1 = 'Urine Ba'
feat_name2 = 'U-Mo'
nick_name2 = 'Urine Mo'
inter1 = pdp.pdp_interact(model=rf, dataset=X_train, model_features=base_features, features=[feat_name1, feat_name2])
fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=[nick_name1, nick_name2])
plt.savefig('fig\\pdp\\Ba2Mo.svg', dpi= 300, format="svg")

feat_name1 = 'U-Ba'
nick_name1 = 'Urine Ba'
feat_name2 = 'B-Cd'
nick_name2 = 'Blood Cd'
inter1 = pdp.pdp_interact(model=rf, dataset=X_train, model_features=base_features, features=[feat_name1, feat_name2])
fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=[nick_name1, nick_name2])
plt.savefig('fig\\pdp\\Ba2Cd.svg', dpi= 300, format="svg")

feat_name1 = 'U-Ba'
nick_name1 = 'Urine Ba'
feat_name2 = 'B-Pb'
nick_name2 = 'Blood Pb'
inter1 = pdp.pdp_interact(model=rf, dataset=X_train, model_features=base_features, features=[feat_name1, feat_name2])
fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=[nick_name1, nick_name2])
plt.savefig('fig\\pdp\\Ba2Pb.svg', dpi= 300, format="svg")

feat_name1 = 'U-Mo'
nick_name1 = 'Urine Mo'
feat_name2 = 'B-Pb'
nick_name2 = 'Blood Pb'
inter1 = pdp.pdp_interact(model=rf, dataset=X_train, model_features=base_features, features=[feat_name1, feat_name2])
fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=[nick_name1, nick_name2])
plt.savefig('fig\\pdp\\Mo2Pb.svg', dpi= 300, format="svg")

feat_name1 = 'U-Mo'
nick_name1 = 'Urine Mo'
feat_name2 = 'B-Cd'
nick_name2 = 'Blood Cd'
inter1 = pdp.pdp_interact(model=rf, dataset=X_train, model_features=base_features, features=[feat_name1, feat_name2])
fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=[nick_name1, nick_name2])
plt.savefig('fig\\pdp\\Mo2Cd.svg', dpi= 300, format="svg")

feat_name1 = 'B-Pb'
nick_name1 = 'Blood Pb'
feat_name2 = 'B-Cd'
nick_name2 = 'Blood Cd'
inter1 = pdp.pdp_interact(model=rf, dataset=X_train, model_features=base_features, features=[feat_name1, feat_name2])
fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=[nick_name1, nick_name2])
plt.savefig('fig\\pdp\\Pb2Cd.svg', dpi= 300, format="svg")


# In[ ]:


# SHAP
shap.initjs()
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
expected_value = explainer.expected_value
shap.decision_plot(expected_value[1], shap_values[1], X_test, alpha = 0.3, new_base_value=0.5,show=False)
plt.savefig('fig\\shap\\decision.svg', dpi= 300, format="svg")


# In[ ]:


idx = 9
patient = X_test.iloc[idx,:]
shap_values_patient = explainer.shap_values(patient)
shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_values_patient[1], patient,show=False)
plt.savefig('fig\\shap\\waterfall.svg', dpi= 300, format="svg")

