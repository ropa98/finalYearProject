#!/usr/bin/env python
# coding: utf-8
import random
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

SEED_VAL = 1000
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)

pd.set_option('display.max_columns', None)

targets = pd.read_csv('train.csv')
payment_history = pd.read_csv('payment_history.csv')
client_data = pd.read_csv('client_data.csv')
policy_data = pd.read_csv('policy_data.csv')
sample_sub = pd.read_csv('sample_sub.csv')

policy_data.head()

policy_data.columns

policy_data.shape

policy_data.nunique()

policy_data.describe()

print(len(policy_data['Policy ID'].unique()), len(policy_data))

policy_data = policy_data.drop_duplicates()
print(len(policy_data['Policy ID'].unique()), len(policy_data))

policy_data.info()

policy_data['PPR_PRODCD'] = pd.Categorical(policy_data['PPR_PRODCD'])
policy_data['NLO_TYPE'] = pd.Categorical(policy_data['NLO_TYPE'])

dfDummiesProd = pd.get_dummies(policy_data['PPR_PRODCD'], prefix='category')
dfDummiesNLO = pd.get_dummies(policy_data['NLO_TYPE'], prefix='category')

dfDummiesProd['Policy ID'] = policy_data['Policy ID']
dfDummiesNLO['Policy ID'] = policy_data['Policy ID']

dfDummiesProd = dfDummiesProd.groupby(by='Policy ID').sum()
dfDummiesNLO = dfDummiesNLO.groupby(by='Policy ID').sum()

dfDummiesProd.head()

dfDummiesProd.shape

dfDummiesProd.info()

dfDummiesProd.describe()

dfDummiesProd.nunique()

# individual feature plotting on histogram
dfDummiesProd.hist()

# pair plotting of features
import seaborn as sns

sns.pairplot(dfDummiesProd)

# feature correlation
dfDummiesProd.corr(method='pearson')

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))
corrMatrix = dfDummiesProd.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

# Data Visualisation
import seaborn as sns

plt.figure(figsize=(15, 7))
sns.countplot(policy_data['CATEGORY'])
plt.title("CATEGORY COUNT", fontsize=18)
plt.xlabel("CATEGORY", fontsize=15)
plt.ylabel("COUNT", fontsize=15)
plt.show()

# Data Visualisation
import seaborn as sns

plt.figure(figsize=(20, 7))
sns.countplot(policy_data['PPR_PRODCD'])
plt.title("PPR_PRODCD Feature", fontsize=18)
plt.xlabel("PPR_PRODCD", fontsize=15)
plt.ylabel("COUNT", fontsize=15)
plt.show()
# Data Visualisation
import seaborn as sns

plt.figure(figsize=(15, 7))
sns.countplot(policy_data['NLO_TYPE'])
plt.title("NLO_TYPE", fontsize=18)
plt.xlabel("NLO_TYPE", fontsize=15)
plt.ylabel("COUNT", fontsize=15)
plt.show()

# Data Visualisation
import seaborn as sns

plt.figure(figsize=(10, 7))
sns.countplot(policy_data['CLF_LIFECD'])
plt.title("CLF_LIFECD", fontsize=18)
plt.xlabel("CLF_LIFECD", fontsize=15)
plt.ylabel("COUNT", fontsize=15)
plt.show()

# checking the distribution of the target variable
targets['Lapse'].value_counts()

# Checking the distribution of the target variable in percentage
print((targets.groupby('Lapse')['Lapse'].count() / targets['Lapse'].count()) * 100)
((targets.groupby('Lapse')['Lapse'].count() / targets['Lapse'].count()) * 100).plot.pie()

# too null values
policy_data.drop(['NPR_SUMASSURED'], axis=1, inplace=True)

# NLO_AMOUNT - amount if there’s an extra charge
policy_data['NLO_AMOUNT'] = policy_data['NLO_AMOUNT'].fillna(0)
policy_data['NLO_AMOUNT'] = policy_data['NLO_AMOUNT'].apply(lambda x: 1 if x > 0 else 0)

# change date to datetime type
policy_data['NP2_EFFECTDATE'] = policy_data['NP2_EFFECTDATE'].apply(lambda a:
                                                                    datetime(year=int(a.split('/')[2]),
                                                                             month=int(a.split('/')[1]),
                                                                             day=int(a.split('/')[0]))
                                                                    )
policy_data.head()

stats_pdata = policy_data.groupby(by="Policy ID").agg({
    'NP2_EFFECTDATE': ['min', 'max'],
})
stats_pdata.columns = ["_".join(x) for x in stats_pdata.columns.ravel()]
policy_data = pd.merge(policy_data, stats_pdata, on="Policy ID", how="left")
# Add counts
policy_data['count'] = policy_data.groupby(by='Policy ID').transform('count')['NP2_EFFECTDATE']
policy_data.head()

policy_data.drop(['NLO_TYPE', 'PPR_PRODCD'], axis=1, inplace=True)

# last name ~unique identifier , NLO AMount replaced by Nlo_amount_sum
policy_data.drop(['NPH_LASTNAME', 'NLO_AMOUNT'], axis=1, inplace=True)

policy_data['monthOfPolicy'] = policy_data['NP2_EFFECTDATE'].apply(lambda x: x.month)
policy_data['diffMaxMinDate'] = (policy_data['NP2_EFFECTDATE_max'] - policy_data['NP2_EFFECTDATE_min'])
policy_data['diffMaxMinDate'] = policy_data['diffMaxMinDate'].apply(lambda x: x.days)
policy_data['BOOLdiffMaxMinDate'] = policy_data['diffMaxMinDate'].apply(lambda x: 1 if x > 0 else 0)

policy_data = pd.merge(policy_data, dfDummiesNLO, left_on="Policy ID", right_index=True, how="left")
policy_data = pd.merge(policy_data, dfDummiesProd, left_on="Policy ID", right_index=True, how="left")
policy_data.head()

policy_data.drop(['NP2_EFFECTDATE', 'NPR_PREMIUM', 'CLF_LIFECD',
                  'NSP_SUBPROPOSAL', 'NP2_EFFECTDATE_min',
                  'NP2_EFFECTDATE_max'], axis=1, inplace=True)

policy_data = policy_data.drop_duplicates(subset=['Policy ID'])
policy_data.head()

# defining training dataset
train = pd.merge(policy_data, targets[targets.Lapse == '1'], on='Policy ID', how="left")
train['Lapse'] = train.Lapse.fillna(0)
train.drop('Lapse Year', axis=1, inplace=True)

sample_sub.head()

test = pd.DataFrame()
test['Policy ID'] = sample_sub['Policy ID']
test = pd.merge(test, policy_data, how='left', on='Policy ID')
test.head()

le = LabelEncoder()
for i in ['PCL_LOCATCODE', 'OCCUPATION', 'CATEGORY', 'AAG_AGCODE']:
    le.fit(policy_data[i])
    train[i] = le.transform(train[i])
    test[i] = le.transform(test[i])

train.head()

train.rename(columns={'Lapse': 'target'}, inplace=True)

train['target'] = train['target'].astype('int')

train.drop('Policy ID', axis=1, inplace=True)
test.drop('Policy ID', axis=1, inplace=True)

len(train), len(test)

X = train.drop(['target'], axis=1)
y = train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED_VAL)

# In[48]
rf = RandomForestClassifier(n_estimators=400,

                            max_depth=13,
                            random_state=SEED_VAL)
rf.fit(X_train, y_train)
pred = rf.predict_proba(X_test)[:, 1]
print(log_loss(y_test, pred))

# In[49]:


rf = RandomForestClassifier(n_estimators=400,

                            max_depth=13,
                            random_state=SEED_VAL)
rf.fit(X, y)

# In[50]:


preds_rf = rf.predict_proba(test)[:, 1]

# In[51]:


preds_rf

# In[52]:


predictions = rf.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score

print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))

# In[53]:


feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.plot(kind='line')

from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(random_state=0)

# In[55]:


logistic_model.fit(X, y)
preds_logistic = logistic_model.predict_proba(test)[:, 1]
print(log_loss(y_test, pred))


preds_logistic

predictions = logistic_model.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score

print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))

from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=700,
    learning_rate=0.075,
    max_depth=3,
    min_child_weight=12,
    colsample_by_tree=0.7,
    seed=SEED_VAL,
    subsample=1,

)

xgb_model.fit(X_train, y_train)

# In[62]:


pred_xgb = np.array(xgb_model.predict_proba(X_test))[:, 1]

# In[63]:


log_loss(y_test, pred_xgb)

# In[64]:


xgb_model.fit(X, y)
preds_xgb = xgb_model.predict_proba(test)[:, 1]

# In[65]:


preds_xgb

# In[66]:


predictions = xgb_model.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score

print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))

sub = pd.DataFrame()
sub['Policy ID'] = sample_sub['Policy ID']
sub['Lapse'] = preds_rf
sub.head()

# # SAVING THE MODEL

import joblib

joblib.dump(rf, 'customer_churn_prediction_model.joblib')

# # LOADING THE MODEL

model = joblib.load('customer_churn_prediction_model.joblib')

sub.columns
