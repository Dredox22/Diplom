import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import kurtosis, skew
import matplotlib.ticker as ticker
import scipy.stats as st
import random
import scipy
import statsmodels.stats.api as stm
from io import StringIO
import sys
from sklearn import metrics
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

import catboost
from catboost import CatBoostClassifier
from sklearn. pipeline import Pipeline
 
#models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import xgboost as xgb

#scoring
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score, roc_auc_score, precision_recall_curve, roc_curve, auc
 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
 
import shape

import pickle

# Загрузим исходные данные. Общие характеристики полученного датасета.
df = pd.read_csv('megafon.csv', sep=',', index_col=0, encoding='UTF-8')
df.head()

def class_score(Q1):
    if (Q1 == '9') | (Q1 == '10') | (Q1 == '10, 9') | ('Спасибо' in str(Q1)):
        return 'class5'
    elif (Q1 == '6') | (Q1 == '7') | (Q1 == '8') | (Q1 == '5, 7') | (Q1 == '5, 6') |\
            (Q1 == '3, 9') | (Q1 == '10, 5') | (Q1 == '2, 9') | ('Ширяево' in str(Q1)) | ('Бели' in str(Q1)):
        return 'class4'
    elif (Q1 == '3') | (Q1 == '4') | (Q1 == '5') | (Q1 == '2, 5') | (Q1 == '1, 8') | (Q1 == '1, 6') | (Q1 == '3, 7')\
            | ('тройка' in str(Q1)) | ('Заокский' in str(Q1)) | ('ОЦЕНКА-3' in str(Q1)):
        return 'class3'
    elif (Q1 == '0') | (Q1 == '1') | (Q1 == '2') | (Q1 == '1, 3') | (Q1 == '0, 1, 5')\
            | ('Немагу' in str(Q1)) | ('Отвратительно' in str(Q1)) | ('Поохое' in str(Q1)) | ('Ужасно' in str(Q1)):
        return 'class2'
    else:
        return 'class1'


def class_int_score(score):
    if (score == 'class5'):
        return 5
    elif (score == 'class4'):
        return 4
    elif (score == 'class3'):
        return 3
    elif (score == 'class2'):
        return 2
    else:
        return 1


df['class'] = df['Q1'].apply(lambda score: class_score(score))
df['class_rate'] = df['class'].apply(lambda score: class_int_score(score))

df_mean = df.groupby('class').agg('mean')
df_mean = df_mean.drop(index=['class1'])
df_mean

sns.set(rc={'figure.figsize': (11.7, 8.27)}, font_scale=1.2, style='darkgrid')

data = pd.DataFrame(df_mean.drop(['class_rate'], axis=1), df_mean.index)

class_plot = sns.lineplot(data=data, palette='tab10', linewidth=3, alpha=1)

class_plot.set_xlabel('CLASS \n ←  Low       High  →', fontsize=18,
                      fontstyle='normal', color='midnightblue')
class_plot.set_ylabel('Значения параметров\n', fontsize=18, fontfamily='Verdana',
                      fontstyle='normal', color='midnightblue')
class_plot.set_title('Рис.1 Средние метрики качества услуг\n', fontsize=18, fontfamily='Verdana',
                     fontstyle='normal', color='midnightblue')

class_plot.xaxis.set_major_locator(ticker.MultipleLocator(1))

plt.legend(bbox_to_anchor=[1.009, 0.64], loc='center right',
           title='Технические параметры:')
plt.show()

def Q2_with_45(x):
    if ('4' in str(x)) | ('5' in str(x)):
        return 'Q2_with_45'


df['Q2_with_45'] = df['Q2'].apply(lambda x: Q2_with_45(x))
df_Q2_with_45 = df[df['Q2_with_45'] == 'Q2_with_45']
df_Q2_with_45_mean = df_Q2_with_45.groupby('class').agg('mean')
df_Q2_with_45_mean

df= df.drop(['Q1', 'Q2', 'class', 'Q2_with_45'], axis=1)

X = df.drop('class_rate', axis=1)
y = df['class_rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)

model = CatBoostClassifier(
    iterations=150,
    random_seed=43,
    loss_function='MultiClass'
)

model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    verbose=False,
    plot=True
)

#Build pipeline for models
 
pipeline_lr = Pipeline([('lr_classifier',LogisticRegression())])
 
pipeline_dt = Pipeline([('dt_classifier',DecisionTreeClassifier())])
 
pipeline_gbcl = Pipeline([('gbcl_classifier',GradientBoostingClassifier())])
 
pipeline_rf = Pipeline([('rf_classifier',RandomForestClassifier())])
 
pipeline_knn = Pipeline([('knn_classifier',KNeighborsClassifier())])
 
pipeline_bnb = Pipeline([('bnb_classifer',BernoulliNB())])
 
pipeline_bag = Pipeline([('bag_classifer',BaggingClassifier())])
 
pipeline_ada = Pipeline([('bnb_classifer',AdaBoostClassifier())])
 
pipeline_gnb = Pipeline([('gnb_classifer',GaussianNB())])
 
pipeline_mlp = Pipeline([('mlp_classifer',MLPClassifier())])
 
pipeline_sgd = Pipeline([('sgd_classifer',SGDClassifier())])
 
pipeline_xgb = Pipeline([('xgb_classifer',XGBClassifier())])
 
pipeline_cat = Pipeline([('cat_classifier',CatBoostClassifier(verbose=False))])
 
# List of all the pipelines
pipelines = [pipeline_lr, pipeline_dt, pipeline_gbcl, pipeline_rf, pipeline_knn, pipeline_bnb, pipeline_bag, pipeline_ada, pipeline_gnb, pipeline_mlp, pipeline_sgd, pipeline_xgb, pipeline_cat]
 
# Dictionary of pipelines and classifier types for ease of reference
pipe_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'Gradient Boost', 3:'RandomForest', 4: 'KNN', 5: 'BN', 6:'Bagging', 7: 'Ada Boost', 8:'GaussianNB', 9:'MLP Classifier', 10:'SGD Classifier', 11:'XG Boost', 12:'Cat Boost'}
 
 
# Fitting the pipelines
for pipe in pipelines:
    pipe. fit(X_train, y_train-1)
    
cv_results_accuracy = []
for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, X_train, y_train-1, cv=5)
    cv_results_accuracy.append(cv_score)
    print("%s: %f " % (pipe_dict[i], cv_score.mean()))
    
model = CatBoostClassifier(verbose=False)
model. fit(X_train, y_train-1)
 
#Print scores for Multiclass
y_test_pred = model. predict(X_test)
y_test_prob = model.predict_proba(X_test)
 
print(metrics.classification_report(y_test, y_test_pred, digits=3))
print('Accuracy score: ', accuracy_score(y_test, y_test_pred))
print('Roc auc score : ', roc_auc_score(y_test, y_test_prob, multi_class='ovr'))

confusion = confusion_matrix(y_test, y_test_pred)
 
fig = px.imshow(confusion, labels=dict(x="Predicted Value", y="Actual Vlaue"), x=[1,2,3,4,5],y=[1,2,3,4,5] ,text_auto=True, title='Confusion Matrix')
fig.update_layout(title_x=0.5)
fig. show()

model = CatBoostClassifier(
    random_seed=63,
    iterations=200,
    learning_rate=0.05
)

model.fit(
    X_train, y_train,
    verbose=50
)

model.get_feature_importance(prettified=True)

best_model = CatBoostClassifier(iterations=100)
best_model = model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    verbose=False
)

best_model.save_model('catboost_model.json')
best_model.save_model('catboost_model.bin')

best_model.load_model('catboost_model.bin')
print(best_model.get_params())
print(best_model.random_seed_)

fast_model = CatBoostClassifier(
    random_seed=63,
    iterations=150,
    learning_rate=0.01,
    boosting_type='Plain',
    bootstrap_type='Bernoulli',
    subsample=0.5,
    one_hot_max_size=20,
    rsm=0.5,
    leaf_estimation_iterations=5,
    max_ctr_complexity=1
)

fast_model.fit(
    X_train, y_train,
    verbose=False,
    plot=True
)

tunned_model = CatBoostClassifier(
    random_seed=63,
    iterations=1000,
    learning_rate=0.03,
    l2_leaf_reg=3,
    bagging_temperature=1,
    random_strength=1,
    one_hot_max_size=2,
    leaf_estimation_method='Newton' 
)

tunned_model.fit(
    X_train, y_train,
    verbose=False,
    eval_set=(X_test, y_test),
    plot=True

)

best_model = CatBoostClassifier(
    random_seed=63,
    iterations=int(tunned_model.tree_count_ * 1.2)
)

best_model.fit(
    X, y,
    verbose=100
)

predictions=best_model
print(f"Predictions: {predictions}")

print(metrics.classification_report(y_test, y_test_pred, digits=3))
print('Accuracy score: ', accuracy_score(y_test, y_test_pred))
print('Roc auc score : ', roc_auc_score(y_test, y_test_prob, multi_class='ovr'))

# Make Pickle file
pickle.dump(best_model, open("model.pkl", "wb"))