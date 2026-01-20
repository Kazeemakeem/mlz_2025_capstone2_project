#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('breast-cancer.csv')
df.head().T


# In[3]:


df.dtypes


# In[4]:


df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns = df.columns.str.lower().str.replace('-', '_')


# In[5]:


df.columns


# In[6]:


features = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave_points_worst',
       'symmetry_worst', 'fractal_dimension_worst']


# In[7]:


df[features].corr()


# In[8]:


df['diagnosis'].value_counts(normalize=True)


# In[9]:


df.isnull().sum()


# In[10]:


df['diagnosis'] = df['diagnosis'].str.strip()
df['diagnosis'] = (df['diagnosis'] == "M").astype(int)


# In[11]:


df['diagnosis'].value_counts(normalize=True)


# In[12]:


df.corr()


# In[13]:


df.head()


# In[14]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)


# In[15]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[16]:


y_train = df_train.diagnosis.values
y_val = df_val.diagnosis.values
y_test = df_test.diagnosis.values


# In[17]:


del df_train['diagnosis']
del df_val['diagnosis']
del df_test['diagnosis']


# In[18]:


def train_logistic_regression(c_val, features):
    dv = DictVectorizer(sparse=False)
    model = LogisticRegression(solver='liblinear', C=c_val, max_iter=1000, random_state=42)
    
    train_dict = df_train[features].to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    val_dict = df_val[features].to_dict(orient='records')
    X_val = dv.transform(val_dict)
    
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, y_pred)
    auc_pr = auc(recall, precision)
    
    return auc_pr


# In[19]:


train_logistic_regression(1.0, features)


# In[20]:


c = [0.01, 0.1, 1, 10, 100]
for c_val in c:
    reg_acc = train_logistic_regression(c_val, features)
    print(c_val,':', reg_acc)


# DECISION TREE

# In[30]:


def train_dt(depth_max=None, leaf_min=1):
    train_dicts = df_train[features].fillna(0).to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)
    dt = DecisionTreeClassifier(max_depth=depth_max, min_samples_leaf=leaf_min)
    dt.fit(X_train, y_train)
    val_dicts = df_val.fillna(0).to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = dt.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, y_pred)
    auc_pr = auc(recall, precision)
    return auc_pr


# In[24]:


train_dicts = df_train[features].fillna(0).to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)


# In[25]:


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


# In[27]:


val_dicts = df_val[features].fillna(0).to_dict(orient='records')
X_val = dv.transform(val_dicts)


# In[28]:


y_pred = dt.predict_proba(X_val)[:, 1]
precision, recall, _ = precision_recall_curve(y_val, y_pred)
auc_pr = auc(recall, precision)
auc_pr


# In[31]:


depths = [1, 2, 3, 4, 5, 6, 10, 15, 20, None]
dt_scores = []
for d in depths:
    score = train_dt(depth_max=d, leaf_min=1)
    dt_scores.append((d, score))
dt_scores


# In[32]:


tuned_dt_scores = []
for d in [4,5,6]:
    for l in [1, 5, 10, 15, 20, 100, 200, 500]:
        dt = DecisionTreeClassifier(max_depth=d, min_samples_leaf=l)
        dt.fit(X_train, y_train)

        y_pred = dt.predict_proba(X_val)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, y_pred)
        auc_pr = auc(recall, precision)
        
        tuned_dt_scores.append((d, l, auc_pr))


# In[33]:


columns = ['max_depth', 'min_samples_leaf', 'auc']
df_scores = pd.DataFrame(tuned_dt_scores, columns=columns)


# In[34]:


df_scores_pivot = df_scores.pivot(index='min_samples_leaf', columns=['max_depth'], values=['auc'])
df_scores_pivot.round(3)


# In[35]:


sns.heatmap(df_scores_pivot, annot=True, fmt=".4f")


# #### RANDOM FOREST

# In[37]:


rf_scores = []

for n in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators=n, random_state=1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, y_pred)
    auc_pr = auc(recall, precision)
    
    rf_scores.append((n, auc_pr))


# In[38]:


rf_scores


# In[39]:


df_scores = pd.DataFrame(rf_scores, columns=['n_estimators', 'auc_pr'])


# In[40]:


plt.plot(df_scores.n_estimators, df_scores.auc_pr)


# In[41]:


df_scores[df_scores['auc_pr'] == df_scores['auc_pr'].max()]


# In[42]:


tuned_rf_scores = []

for d in [5, 10, 15, 20]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=d,
                                    random_state=1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, y_pred)
        auc_pr = auc(recall, precision)

        tuned_rf_scores.append((d, n, auc_pr))


# In[45]:


columns = ['max_depth', 'n_estimators', 'auc_pr']
df_scores = pd.DataFrame(tuned_rf_scores, columns=columns)
df_scores


# In[44]:


for d in [5, 10, 15, 20]:
    df_subset = df_scores[df_scores.max_depth == d]
    plt.plot(df_subset.n_estimators, df_subset.auc_pr,
             label='max_depth=%d' % d)

plt.legend()


# In[46]:


max_depth = 20
n_estimator = 100


# In[47]:


tuned_rf_scores_2 = []

for s in [1, 3, 5, 10, 50]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=max_depth,
                                    min_samples_leaf=s,
                                    random_state=1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, y_pred)
        auc_pr = auc(recall, precision)

        tuned_rf_scores_2.append((s, n, auc_pr))


# In[48]:


columns = ['min_samples_leaf', 'n_estimators', 'auc_pr']
df_scores = pd.DataFrame(tuned_rf_scores_2, columns=columns)


# In[49]:


colors = ['black', 'blue', 'orange', 'red', 'grey']
values = [1, 3, 5, 10, 50]

for s, col in zip(values, colors):
    df_subset = df_scores[df_scores.min_samples_leaf == s]
    
    plt.plot(df_subset.n_estimators, df_subset.auc_pr,
             color=col,
             label='min_samples_leaf=%d' % s)

plt.legend()


# ### GRADIENT BOOST

# In[50]:


features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)


# In[51]:


neg_count = sum(y_train == 0)
pos_count = sum(y_train == 1)
weight_ratio = neg_count / pos_count

xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr', 
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
    'scale_pos_weight': weight_ratio,
}


model = xgb.train(xgb_params, dtrain, num_boost_round=10)


# In[52]:


y_pred = model.predict(dval)


# In[53]:


precision, recall, _ = precision_recall_curve(y_val, y_pred)
auc_pr = auc(recall, precision)
auc_pr


# In[54]:


watchlist = [(dtrain, 'train'), (dval, 'val')]


# In[55]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.3,\n    'max_depth': 6,\n    'min_child_weight': 1,\n    'objective': 'binary:logistic',\n    'eval_metric': 'aucpr', \n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n    'scale_pos_weight': weight_ratio,\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n")


# In[57]:


def parse_xgb_output(output):
    results = []

    for line in output.stdout.strip().split('\n'):
        it_line, train_line, val_line = line.split('\t')

        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val = float(val_line.split(':')[1])

        results.append((it, train, val))
    
    columns = ['num_iter', 'train_auc_pr', 'val_auc_pr']
    df_results = pd.DataFrame(results, columns=columns)
    return df_results


# In[65]:


eta_scores = {}


# In[67]:


key1 = 'eta=%s' % (xgb_params['eta'])
eta_scores[key1] = parse_xgb_output(output)


# In[68]:


s = output.stdout


# In[69]:


print(s[:200])


# In[70]:


df_score = parse_xgb_output(output)


# In[71]:


plt.plot(df_score.num_iter, df_score.train_auc_pr, label='train')
plt.plot(df_score.num_iter, df_score.val_auc_pr, label='val')
plt.legend()


# In[72]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.01, \n    'max_depth': 6,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'aucpr',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n    'scale_pos_weight': weight_ratio, \n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n")


# In[73]:


key2 = 'eta=%s' % (xgb_params['eta'])
eta_scores[key2] = parse_xgb_output(output)


# In[74]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.1, \n    'max_depth': 6,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'aucpr',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n    'scale_pos_weight': weight_ratio, \n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n")


# In[75]:


key3 = 'eta=%s' % (xgb_params['eta'])
eta_scores[key3] = parse_xgb_output(output)


# In[76]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.3, \n    'max_depth': 6,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'aucpr',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n    'scale_pos_weight': weight_ratio, \n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n")


# In[77]:


key4 = 'eta=%s' % (xgb_params['eta'])
eta_scores[key4] = parse_xgb_output(output)


# In[78]:


etas = ['eta=0.1', 'eta=0.01', 'eta=0.3']
for eta in etas:
    df_score = eta_scores[eta]
    plt.plot(df_score.num_iter, df_score.val_auc_pr, label=eta)
    plt.legend()


# #### TUNE MAX_DEPTH

# In[111]:


depth_scores = {}


# In[112]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.3, \n    'max_depth': 3,\n    'min_child_weight': 20,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'aucpr',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n    'scale_pos_weight': weight_ratio, \n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n")


# In[113]:


key1 = 'max_depth=%s' % (xgb_params['max_depth'])
depth_scores[key1] = parse_xgb_output(output)


# In[114]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.3, \n    'max_depth': 4,\n    'min_child_weight': 20,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'aucpr',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n    'scale_pos_weight': weight_ratio, \n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n")


# In[115]:


key2 = 'max_depth=%s' % (xgb_params['max_depth'])
depth_scores[key2] = parse_xgb_output(output)


# In[116]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.3, \n    'max_depth': 6,\n    'min_child_weight': 20,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'aucpr',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n    'scale_pos_weight': weight_ratio, \n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n")


# In[117]:


key3 = 'max_depth=%s' % (xgb_params['max_depth'])
depth_scores[key3] = parse_xgb_output(output)


# In[118]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.3, \n    'max_depth': 10,\n    'min_child_weight': 20,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'aucpr',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n    'scale_pos_weight': weight_ratio, \n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n")


# In[119]:


key4 = 'max_depth=%s' % (xgb_params['max_depth'])
depth_scores[key4] = parse_xgb_output(output)


# In[120]:


print(depth_scores.items())
for max_depth, df_score in depth_scores.items():
    plt.plot(df_score.num_iter, df_score.val_auc_pr, label=max_depth)

# plt.ylim(0.8, 0.84)
plt.legend()


# In[95]:


cw_scores = {}


# In[121]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.3, \n    'max_depth': 3,\n    'min_child_weight': 1,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'aucpr',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n    'scale_pos_weight': weight_ratio, \n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n")


# In[122]:


key1 = 'min_child_weight=%s' % (xgb_params['min_child_weight'])
cw_scores[key1] = parse_xgb_output(output)


# In[123]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.3, \n    'max_depth': 3,\n    'min_child_weight': 10,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'aucpr',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n    'scale_pos_weight': weight_ratio, \n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n")


# In[124]:


key2 = 'min_child_weight=%s' % (xgb_params['min_child_weight'])
cw_scores[key2] = parse_xgb_output(output)


# In[125]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.3, \n    'max_depth': 3,\n    'min_child_weight': 20,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'aucpr',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n    'scale_pos_weight': weight_ratio, \n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n")


# In[126]:


key3 = 'min_child_weight=%s' % (xgb_params['min_child_weight'])
cw_scores[key3] = parse_xgb_output(output)


# In[127]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.3, \n    'max_depth': 3,\n    'min_child_weight': 30,\n    \n    'objective': 'binary:logistic',\n    'eval_metric': 'aucpr',\n\n    'nthread': 8,\n    'seed': 1,\n    'verbosity': 1,\n    'scale_pos_weight': weight_ratio, \n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n")


# In[128]:


key4 = 'min_child_weight=%s' % (xgb_params['min_child_weight'])
cw_scores[key4] = parse_xgb_output(output)


# In[129]:


for min_child_weight, df_score in cw_scores.items():
    plt.plot(df_score.num_iter, df_score.val_auc_pr, label=min_child_weight)
plt.legend()


# ### CHOOSING THE BEST MODEL

# #### Logistic Regression

# In[105]:


train_logistic_regression(0.1, features)


# #### Decision Tree

# In[106]:


train_dt(5,20)


# #### Random Forest

# In[110]:


rf = RandomForestClassifier(n_estimators=100,
                            max_depth=20,
                            min_samples_leaf=3,
                            random_state=1)
rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_val)[:, 1]
precision, recall, _ = precision_recall_curve(y_val, y_pred)
auc_pr = auc(recall, precision)
auc_pr


# #### Gradient Boost

# In[130]:


final_xgb_params = {
    'eta': 0.3, 
    'max_depth': 3,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
    'scale_pos_weight': weight_ratio, 
}

model = xgb.train(final_xgb_params, dtrain, num_boost_round=200)

y_pred = model.predict(dval)
precision, recall, _ = precision_recall_curve(y_val, y_pred)
auc_pr = auc(recall, precision)
auc_pr


# ### FINAL MODEL - RANDOM FOREST

# In[131]:


df_full_train = df_full_train.reset_index(drop=True)


# In[133]:


y_full_train = df_full_train.diagnosis.values


# In[134]:


del df_full_train['diagnosis']


# In[135]:


train_dicts = df_full_train[features].fillna(0).to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(train_dicts)


# In[136]:


test_dicts = df_test[features].fillna(0).to_dict(orient='records')
X_test = dv.transform(test_dicts)


# In[138]:


rf = RandomForestClassifier(n_estimators=100,
                            max_depth=20,
                            min_samples_leaf=3,
                            random_state=1)
rf.fit(X_full_train, y_full_train)
y_pred = rf.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_pred)
auc_pr = auc(recall, precision)
auc_pr


# In[ ]:




