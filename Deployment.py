!pip install pandas
!pip install numpy
!pip install sklearn
!pip install matplotlib
!pip install seaborn
!pip install statsmodels
!pip install randomForest
!pip install lime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from random import seed
from random import randint
from sklearn.utils import resample

diabetes_data = pd.read_csv('diabetes_data.csv', sep=';')

print(diabetes_data.info())
print(diabetes_data.isna().sum())

selected_vars = ["gender", "polyuria", "polydipsia", "sudden_weight_loss", "polyphagia", "irritability", "partial_paresis", "class"]
df_diabetes = diabetes_data[selected_vars]
df_diabetes['gender'] = df_diabetes['gender'].map({0: 'Male', 1: 'Female'})

print(df_diabetes.head())
print(df_diabetes.isna().sum())

seed(1)

train_data_full, test_data_full = train_test_split(df_diabetes, test_size=0.2, random_state=1)

X_train_full = train_data_full.drop('class', axis=1)
y_train_full = train_data_full['class']
X_test_full = test_data_full.drop('class', axis=1)
y_test_full = test_data_full['class']

model_1 = sm.Logit(y_train_full, sm.add_constant(X_train_full)).fit()
print(model_1.summary())

predictions_1 = model_1.predict(sm.add_constant(X_test_full))
predicted_class_1 = np.where(predictions_1 > 0.5, 1, 0)

confusionMatrix_1 = confusion_matrix(y_test_full, predicted_class_1)
auc_1 = roc_auc_score(y_test_full, predictions_1)

selected_vars_2 = ["gender", "polyuria", "polydipsia", "sudden_weight_loss", "irritability", "class"]

train_data_2 = train_data_full[selected_vars_2]
test_data_2 = test_data_full[selected_vars_2]

model_2 = sm.Logit(y_train_full, sm.add_constant(train_data_2)).fit()
print(model_2.summary())

predictions_2 = model_2.predict(sm.add_constant(test_data_2))
predicted_class_2 = np.where(predictions_2 > 0.5, 1, 0)

confusionMatrix_2 = confusion_matrix(y_test_full, predicted_class_2)
auc_2 = roc_auc_score(y_test_full, predictions_2)

print(confusionMatrix_1)
print(auc_1)
print(confusionMatrix_2)
print(auc_2)

seed(2)
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)

cv_results_1 = []
for train_index, test_index in folds.split(df_diabetes[selected_vars], df_diabetes['class']):
    train_fold = df_diabetes[selected_vars].iloc[train_index]
    test_fold = df_diabetes[selected_vars].iloc[test_index]
    model_fold = sm.Logit(train_fold['class'], sm.add_constant(train_fold.drop('class', axis=1))).fit()
    predictions_fold = model_fold.predict(sm.add_constant(test_fold.drop('class', axis=1)))
    roc_auc = roc_auc_score(test_fold['class'], predictions_fold)
    cv_results_1.append(roc_auc)
mean_auc_1 = np.mean(cv_results_1)
print(mean_auc_1)

cv_results_2 = []
for train_index, test_index in folds.split(df_diabetes[selected_vars_2], df_diabetes['class']):
    train_fold = df_diabetes[selected_vars_2].iloc[train_index]
    test_fold = df_diabetes[selected_vars_2].iloc[test_index]
    model_fold = sm.Logit(train_fold['class'], sm.add_constant(train_fold.drop('class', axis=1))).fit()
    predictions_fold = model_fold.predict(sm.add_constant(test_fold.drop('class', axis=1)))
    roc_auc = roc_auc_score(test_fold['class'], predictions_fold)
    cv_results_2.append(roc_auc)
mean_auc_2 = np.mean(cv_results_2)
print(mean_auc_2)

train_data_full['class'] = train_data_full['class'].astype('category')
test_data_full['class'] = test_data_full['class'].astype('category')
train_data_2['class'] = train_data_2['class'].astype('category')
test_data_2['class'] = test_data_2['class'].astype('category')

model_rf_full = RandomForestClassifier(n_estimators=500, random_state=0)
model_rf_full.fit(X_train_full, y_train_full)
print(model_rf_full)

predictions_rf_full = model_rf_full.predict(X_test_full)
confusionMatrix_rf_full = confusion_matrix(y_test_full, predictions_rf_full)
print(confusionMatrix_rf_full)

model_rf_simplified = RandomForestClassifier(n_estimators=500, random_state=0)
model_rf_simplified.fit(train_data_2.drop('class', axis=1), train_data_2['class'])
print(model_rf_simplified)

predictions_rf_simplified = model_rf_simplified.predict(test_data_2.drop('class', axis=1))
confusionMatrix_rf_simplified = confusion_matrix(test_data_2['class'], predictions_rf_simplified)
print(confusionMatrix_rf_simplified)

model_rf_full = RandomForestClassifier(n_estimators=600, random_state=0)
model_rf_full.fit(X_train_full, y_train_full)
print(model_rf_full)

predictions_rf_full = model_rf_full.predict(X_test_full)
confusionMatrix_rf_full = confusion_matrix(y_test_full, predictions_rf_full)
print(confusionMatrix_rf_full)

model_rf_simplified = RandomForestClassifier(n_estimators=600, random_state=0)
model_rf_simplified.fit(train_data_2.drop('class', axis=1), train_data_2['class'])
print(model_rf_simplified)

predictions_rf_simplified = model_rf_simplified.predict(test_data_2.drop('class', axis=1))
confusionMatrix_rf_simplified = confusion_matrix(test_data_2['class'], predictions_rf_simplified)
print(confusionMatrix_rf_simplified)

mtry_values_full = [1, 3, 5, 7]
results_full = {}
for mtry_val in mtry_values_full:
    seed(2)
    model_rf_full = RandomForestClassifier(n_estimators=500, random_state=0, max_features=mtry_val)
    model_rf_full.fit(X_train_full, y_train_full)
    predictions_rf_full = model_rf_full.predict(X_test_full)
    confusionMatrix_rf_full = confusion_matrix(y_test_full, predictions_rf_full)
    results_full["mtry = " + str(mtry_val)] = {"Model": model_rf_full, "ConfusionMatrix": confusionMatrix_rf_full}
for result_name, result in results_full.items():
    print(result_name)
    print(result["Model"])
    print(result["ConfusionMatrix"])
    print()

mtry_values_simplified = [1, 3, 5]
results_simplified = {}
for mtry_val in mtry_values_simplified:
    seed(2)
    model_rf_simplified = RandomForestClassifier(n_estimators=500, random_state=0, max_features=mtry_val)
    model_rf_simplified.fit(train_data_2.drop('class', axis=1), train_data_2['class'])
    predictions_rf_simplified = model_rf_simplified.predict(test_data_2.drop('class', axis=1))
    confusionMatrix_rf_simplified = confusion_matrix(test_data_2['class'], predictions_rf_simplified)
    results_simplified["mtry = " + str(mtry_val)] = {"Model": model_rf_simplified, "ConfusionMatrix": confusionMatrix_rf_simplified}
for result_name, result in results_simplified.items():
    print(result_name)
    print(result["Model"])
    print(result["ConfusionMatrix"])
    print()

mtry_values_full = [1, 3, 5, 7]
results_full = {}
for mtry_val in mtry_values_full:
    seed(2)
    model_rf_full = RandomForestClassifier(n_estimators=600, random_state=0, max_features=mtry_val)
    model_rf_full.fit(X_train_full, y_train_full)
    predictions_rf_full = model_rf_full.predict(X_test_full)
    confusionMatrix_rf_full = confusion_matrix(y_test_full, predictions_rf_full)
    results_full["mtry = " + str(mtry_val)] = {"Model": model_rf_full, "ConfusionMatrix": confusionMatrix_rf_full}
for result_name, result in results_full.items():
    print(result_name)
    print(result["Model"])
    print(result["ConfusionMatrix"])
    print()

mtry_values_simplified = [1, 3, 5]
results_simplified = {}
for mtry_val in mtry_values_simplified:
    seed(123)
    model_rf_simplified = RandomForestClassifier(n_estimators=600, random_state=0, max_features=mtry_val)
    model_rf_simplified.fit(train_data_2.drop('class', axis=1), train_data_2['class'])
    predictions_rf_simplified = model_rf_simplified.predict(test_data_2.drop('class', axis=1))
    confusionMatrix_rf_simplified = confusion_matrix(test_data_2['class'], predictions_rf_simplified)
    results_simplified["mtry = " + str(mtry_val)] = {"Model": model_rf_simplified, "ConfusionMatrix": confusionMatrix_rf_simplified}
for result_name, result in results_simplified.items():
    print(result_name)
    print(result["Model"])
    print(result["ConfusionMatrix"])
    print()

importance_rf = pd.DataFrame({'Feature': X_train_full.columns, 'Importance': model_rf_full.feature_importances_})
importance_rf = importance_rf.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_rf)
plt.title('Feature Importance in Random Forest Model')
plt.show()



