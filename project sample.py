# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.ensemble import SMOTE
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve

# Step 2: Load the dataset
information = pd.read_csv('liver_disorders.csv')

# Step 3: Clean the dataset
information = information.dropna(axis=0, how='any')
information = information.bfill()

# Step 4: Exploratory Data Analysis
numeric_columns = information.select_dtypes(include=['int64', 'float64']).columns
for column in numeric_columns:
    plt.hist(information[column], bins=30)
    plt.title(column)
    plt.show()

sns.pairplot(information, vars=numeric_columns)

corr_matrix = information.corr()
sns.heatmap(corr_matrix, annot=True)

# Step 5: Prepare the dataset for modeling
X = information.drop('Result', axis=1)
y_result = information['Result']
y_disorder = information['disorder']

# Step 6: Feature selection
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
X = X[numeric_columns]

selector = LabelEncoder()
X['disorder'] = selector.fit_transform(X['disorder'])

numeric_features = X.columns
selector = SelectKBest(k=10, score_func=f_classif)
X_new = selector.fit_transform(X, y_disorder)
X_new_df = pd.DataFrame(X_new, columns=numeric_features)

# Step 7: Data scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_new_df)

# Step 8: Split the dataset into training and testing sets
X_train, X_test, y_result_train, y_result_test = train_test_split(X_scaled, y_result, test_size=0.2, random_state=42)

# Step 9: Train and evaluate the model for 'Result' prediction
best_model_result = GridSearchCV(MLPClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
best_model_result.fit(X_train, y_result_train)

# Step 10: Train and evaluate the model for 'disorder' prediction
mlp_classifier_disorder = MLPClassifier(random_state=42)
mlp_classifier_disorder.fit(X_train, y_disorder_train)

# Step 11: Calculate ROC AUC scores for 'Result' prediction
y_result_scores_train = best_model_result.predict_proba(X_train)[:, 1]
y_result_scores_test = best_model_result.predict_proba(X_test)[:, 1]
auc_train_result = roc_auc_score(y_result_train, y_result_scores_train)
auc_test_result = roc_auc_score(y_result_test, y_result_scores_test)

# Step 12: Calculate ROC AUC scores for 'disorder' prediction
y_disorder_scores_train = mlp_classifier_disorder.predict_proba(X_train)[:, 0]
y_disorder_scores_test = mlp_classifier_disorder.predict_proba(X_test)[:, 0]
class_labels = pd.