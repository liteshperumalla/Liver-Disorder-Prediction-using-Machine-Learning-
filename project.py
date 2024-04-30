# Step -1: Importing the Necessary modeules 1. Numpy, 2. Pandas, 3. Sckit-Learn, 4. Tensorflow.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense
from scipy import stats
from sklearn.preprocessing import LabelEncoder,label_binarize
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler


#Step -2: Importing the Dataset csv file
information = pd.read_csv('/Users/liteshperumalla/Desktop/Files/projects/archive/dataset1.csv')
"""pd.options.display.max_rows"""
# Top 5 Rows in the Dataset
print(information.head())
# Total Columns in the Dataset.
"""print(information.to_string())
print(information.info())"""
# Counting the number of Liver Disorder patients
result = information['Result'].value_counts()
count = result.get(1, 0)
print(f"Number of liver patients  detected: {count}")
count = result.get(2, 0)
print(f"Number of no liver patients  detected: {count}")
#Number of liver patients  detected: 21917
#Number of no liver patients  detected: 8774
# Info  of Instaces / Columns in the Dataset
print(information.shape)
#Step -3: Data Cleaning : Checking for the Null values in the Dataset.
print(information.isnull().sum())
#Replacing Null values with the Mean value
#Using the bfill method to fill values by backwards.
information= information.bfill()
print(information.isnull().sum())
#Converting the Categorical datatype to numerical Datatype.
#Looking for the categorical variables
print(information.columns)
# FInding the column in the Dataset
print('Gender ' in information)
#changing the datatype using the astype method in pandas.
information['Gender '] = information['Gender '].astype('category').cat.codes
#Checking for changes
print(information.info())
#Gives the statistics of the DataFrame.
print(information.describe())
#Checking the Duplicate Values
print(information.drop_duplicates())
print(information.info())
# Previous there are 30691
#After Dropping the rows there are 30691 so, there are no Duplicate values.
#exploratory Data Analysis
numeric_columns = information.select_dtypes(include=['float64', 'int64']).columns
for column in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(information[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()
sns.pairplot(information)
plt.show()
correlation_matrix = information.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
categorical_columns = information.select_dtypes(include='object').columns
for cat_column in categorical_columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=cat_column, y='target_variable', data=information)
    plt.title(f'Box plot of {cat_column} vs. target_variable')
    plt.show()
#Checkin for the outliers
z_scores = stats.zscore(information[['age ','Total_Bilirubin', 'Direct_Bilirubin', 'Alkphos_Alkaline_Phosphotase', 'Sgpt_Alamine_Aminotransferase', 'Sgot_Aspartate_Aminotransferase', 'Total_Protiens', 'ALB_Albumin', 'A/G_Ratio_Albumin_and_Globulin_Ratio']])
abs_z_scores = np.abs(z_scores)
threshold = 3
outliers = abs(z_scores) > threshold
print("Outliers:")
#Identified Outliers shown as True
print(outliers)
#Removing the Outliers
new_information = information[(np.abs(stats.zscore(information.select_dtypes(exclude='object'))) < 1).all(axis=1)]
print(new_information)
# Totla number of rows are 3832
#Converting datavalues of result from 2 to 0 and 1 to 1
information['Result'] = information['Result'].map({2:0,1:1})
print(information)
#Identifying the type of Disorders
"""Types of Disorders
1.Jaundice (Bilirubin is higher).
2.Gilbert's Syndrome.
3.Chirrosis.
4. None
5. None
6. Dehydration, Chronic Liver Diseases.
7.Hepatitis
8.cirrhosis, hepatitis, and fatty liver disease.
9.Jaundice, Dehydration, Chronic Inflammatory Conditions.
10. Not required
Hepatitis caused when elevated sgot_AST sgpt_ALT and bilirubin (1,2,3,4,5)
Chirrosis casued when elevated sgot_AST sgpt_ALT bilirubin and low Albumin (1,2,3,4,5,7)
Fatty liver disease when elevated sgot_AST sgpt_ALT and Alkphos_Alkaline_Phosphotase (3,4,5,7)
"""
information['disorder'] = ' '
information.loc[(information['Result'] == 1) & (information['Total_Bilirubin'] > 1.2), 'disorder'] += '1'
information.loc[(information['Result'] == 1) & (information['Direct_Bilirubin'] > 0.3), 'disorder'] += '2'
information.loc[(information['Result'] == 1) & (information['Alkphos_Alkaline_Phosphotase'] > 140), 'disorder'] += '3'
information.loc[(information['Result'] == 1) & (information['Sgpt_Alamine_Aminotransferase'] > 56), 'disorder'] += '4'
information.loc[(information['Result'] == 1) & (information['Sgot_Aspartate_Aminotransferase'] > 40), 'disorder'] += '5'
information.loc[(information['Result'] == 1) & (information['Total_Protiens'] > 8.3), 'disorder'] += '6'
information.loc[(information['Result'] == 1) & (information['ALB_Albumin'] < 5.5), 'disorder'] += '7'
information.loc[(information['Result'] == 1) & (information['ALB_Albumin'] > 5.5), 'disorder'] += '8'
information.loc[(information['Result'] == 1) & (information['A/G_Ratio_Albumin_and_Globulin_Ratio'] > 2.2), 'disorder'] += '9'
result_1_disorders = information.loc[information['Result'] == 1, 'disorder'].value_counts()
print(result_1_disorders)
print(information['disorder'].unique())
information.loc[information['Result'] == '0', 'disorder'] = 'No disorder'
#Categorizing the Disorders
disorder_map = {
    ' 123457': 'Hepatitis',
    ' 7': 'Hepatitis',
    ' 12357': 'Cirrhosis',
    ' 1237': 'Cirrhosis',
    ' 3457': 'Fatty Liver Disease',
    ' 357': 'Fatty Liver Disease',
    ' 37': 'Chronic Liver Diseases',
    ' 237': 'Liver Tumors',
    ' 2357': 'Liver Tumors',
    ' 123579': 'Jaundice',
    ' 1234579': 'Jaundice',
    ' 12579': 'Jaundice',
    ' 23579': 'Jaundice',
    ' 1347' : 'Hepatitis',
    ' 1234567': 'Cirrhosis',
    ' 1234579': 'Cirrhosis',
    ' 1234579': 'Cirrhosis',
    ' 3567': 'Liver Disease',
    ' 367': 'Liver Disease',
}
information['disorder'] = information['disorder'].apply(lambda x: disorder_map.get(x, 'Not Disorder'))
information.to_csv('/Users/liteshperumalla/Desktop/Files/projects/archive/output.csv')
num_unique_disorders = information['disorder'].nunique()
print("Number of different disorders:", num_unique_disorders)
disorder_counts = information['disorder'].value_counts()
print("Types of disorders and their frequencies:")
for disorder, count in disorder_counts.items():
    print(f"{disorder}: {count}")
#changing Data type to integer
information.info()
label_encoder = LabelEncoder()
y_disorder_encoded = label_encoder.fit_transform(information['disorder'])
"""# Feature Engineering
# Example: Creating polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(information.drop(['Result', 'disorder'], axis=1))

# Dimensionality Reduction
pca = PCA(n_components=10)  # Choose the number of principal components
X_pca = pca.fit_transform(X_poly)

# Feature Selection
# Example: Selecting top k features based on ANOVA F-value
selector = SelectKBest(score_func=f_classif, k=10)  # Choose the number of features to select
X_selected = selector.fit_transform(X_pca, information['Result'])"""


#min-max Scaling
# Select only numeric columns for min-max scaling
numeric_columns = information.select_dtypes(include=['float64', 'int64']).columns
# Perform min-max scaling on numeric columns
for column in numeric_columns:
    information[column] = (information[column] - information[column].min()) / (information[column].max() - information[column].min())
print(information.head())

X = information.iloc[:, :-2]
y_result = information['Result']
y_disorder = information['disorder']
# Feature Engineering
# Selecting top k features based on ANOVA F-value for 'Result' prediction
selector_result = SelectKBest(score_func=f_classif, k=10) 
X_selected_result = selector_result.fit_transform(X, y_result)

# Selecting top k features based on ANOVA F-value for 'disorder' prediction
selector_disorder = SelectKBest(score_func=f_classif, k=10)
X_selected_disorder = selector_disorder.fit_transform(X, y_disorder)
# Feature Scaling for 'Result' prediction
scaler_result = StandardScaler()
X_scaled_result = scaler_result.fit_transform(X_selected_result)

# Feature Scaling for 'disorder' prediction
scaler_disorder = StandardScaler()
X_scaled_disorder = scaler_disorder.fit_transform(X_selected_disorder)
# Calculate the distribution of class labels
class_distribution = information['disorder'].value_counts()

# Visualize class distribution using a bar plot
plt.figure(figsize=(10, 6))
class_distribution.plot(kind='bar')
plt.xlabel('Disorder Class')
plt.ylabel('Frequency')
plt.title('Class Distribution')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# Print class distribution
print("Class Distribution:")
print(class_distribution)
selector_result = SelectKBest(score_func=f_classif, k=10) 
X_selected_result = selector_result.fit_transform(X, y_result)

# Selecting top k features based on ANOVA F-value for 'disorder' prediction
selector_disorder = SelectKBest(score_func=f_classif, k=10)
X_selected_disorder = selector_disorder.fit_transform(X, y_disorder)
# Split the data into train and test sets
X_train, X_test, y_result_train, y_result_test, y_disorder_train, y_disorder_test = train_test_split(X, y_result, y_disorder, test_size=0.2, random_state=42)
# Feature Scaling for 'Result' prediction
scaler_result = StandardScaler()
X_scaled_result = scaler_result.fit_transform(X_selected_result)

# Feature Scaling for 'disorder' prediction
scaler_disorder = StandardScaler()
X_scaled_disorder = scaler_disorder.fit_transform(X_selected_disorder)
# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_result_train_resampled = smote.fit_resample(X_train, y_result_train)

# Initialize the MLPClassifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(300, 300), max_iter=1000)

# Fit the classifier to the balanced data
mlp_classifier.fit(X_train_resampled, y_result_train_resampled)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
classifiers = {
    'Neural Network': MLPClassifier(hidden_layer_sizes=(300, 300), max_iter=1000)
}
def create_mlp_classifier():
    return MLPClassifier(hidden_layer_sizes=(50,50), max_iter=1000)
# Cross-validate for 'Result' prediction
mlp_classifier = create_mlp_classifier()
#mlp_classifier.fit(X_train_resampled, y_result_train_resampled, class_weight=class_weights)
cv_results = cross_validate(mlp_classifier, X_train, y_result_train, cv=5, scoring='accuracy', return_train_score=False)
print("Cross-Validation Scores for 'Result' prediction:", cv_results)

# Grid search for hyperparameters
param_grid = {
    'hidden_layer_sizes': [(100,)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001],
    'learning_rate': ['constant'],
     'max_iter': [1000],
     'random_state': [42],
}
grid_search = GridSearchCV(mlp_classifier, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_result_train)
print("Best Hyperparameters for 'Result' prediction:", grid_search.best_params_)
mean_accuracy = np.mean(grid_search.cv_results_['mean_test_score'])
print(f"Mean Accuracy for 'Result' prediction: {mean_accuracy:.2f}")
# Hyperparameter Tuning for 'disorder' prediction
param_grid_disorder = {
    'hidden_layer_sizes': [(100,)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001],
    'learning_rate': ['constant'],
     'max_iter': [1000],
     'random_state': [42],
}
grid_search_disorder = GridSearchCV(mlp_classifier, param_grid_disorder, cv=5, scoring='accuracy')
grid_search_disorder.fit(X_scaled_disorder, y_disorder)


# Train and evaluate model for 'Result' prediction
best_model_result = grid_search.best_estimator_
best_model_result.fit(X_train, y_result_train)
y_result_pred = best_model_result.predict(X_test)
accuracy_result = accuracy_score(y_result_test, y_result_pred)
report_result = classification_report(y_result_test, y_result_pred)
print("Evaluation for 'Result' prediction:")
print(f"Accuracy: {accuracy_result:.2f}")
print(report_result)

# Train and evaluate model for 'disorder' prediction
mlp_classifier_disorder = create_mlp_classifier()
mlp_classifier_disorder.fit(X_train, y_disorder_train)
y_disorder_pred = mlp_classifier_disorder.predict(X_test)
accuracy_disorder = accuracy_score(y_disorder_test, y_disorder_pred)
report_disorder = classification_report(y_disorder_test, y_disorder_pred)
print("Evaluation for 'disorder' prediction:")
print(f"Accuracy: {accuracy_disorder:.2f}")
print(report_disorder)

# Calculate ROC AUC scores for 'Result' prediction
y_result_scores_train = best_model_result.predict_proba(X_train)[:, 1]
y_result_scores_test = best_model_result.predict_proba(X_test)[:, 1]
auc_train_result = roc_auc_score(y_result_train, y_result_scores_train)
auc_test_result = roc_auc_score(y_result_test, y_result_scores_test)
print("ROC AUC scores for 'Result' prediction:")
print(f'Train ROC AUC = {auc_train_result:.3f}, Test ROC AUC = {auc_test_result:.3f}')
# Print other relevant attributes and parameters as needed
fpr_result, tpr_result, _ = roc_curve(y_result_test, y_result_scores_test)
roc_auc_result = auc(fpr_result, tpr_result)

plt.figure(figsize=(8, 6))
plt.plot(fpr_result, tpr_result, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc_result:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Result Prediction')
plt.legend()
plt.show()
# Calculate ROC AUC scores for 'Disorder' prediction
y_disorder_scores_train = mlp_classifier_disorder.predict_proba(X_train)[:, 1]
print(y_disorder_scores_train.shape)
y_disorder_scores_train = y_disorder_scores_train.reshape(-1, 1)
y_disorder_scores_train_proba = mlp_classifier_disorder.predict_proba(X_train)
# Ensure probability estimates sum up to 1 over classes
y_disorder_scores_train_proba /= y_disorder_scores_train_proba.sum(axis=1)[:, np.newaxis]
auc_train_disorder = roc_auc_score(y_disorder_train, y_disorder_scores_train_proba, multi_class='ovo')
y_disorder_scores_test = mlp_classifier_disorder.predict_proba(X_test)
print(y_disorder_scores_test)
auc_test_disorder = roc_auc_score(y_disorder_test, y_disorder_scores_test, multi_class='ovo')
y_disorder_scores_train = mlp_classifier_disorder.predict_proba(X_train)
class_labels = pd.Series(y_disorder_test).unique()
print("Class Labels:", class_labels)
y_disorder_test_bin = label_binarize(y_disorder_test, classes=class_labels)
print("Shape of y_disorder_test_bin:", y_disorder_test_bin.shape)
print("Shape of y_disorder_scores_test:", y_disorder_scores_test.shape)
print("Shape of y_disorder_test_bin:", y_disorder_test_bin.shape)
print("Unique values in y_disorder_test_bin:", np.unique(y_disorder_test_bin))
# Debugging: Print out the arrays for inspection
print("y_disorder_scores_test:", y_disorder_scores_test)
print("y_disorder_test_bin:", y_disorder_test_bin)
class_accuracies = {}

# Iterate over each class label
for i, class_label in enumerate(class_labels):
    # Find the indices where the true label matches the current class label
    class_indices = np.where(y_disorder_test == class_label)[0]

    # Get the predicted probabilities and true labels for samples belonging to this class
    class_predictions = y_disorder_scores_test[class_indices, i]
    true_labels = y_disorder_test_bin[class_indices, i]

    # Threshold the predicted probabilities to obtain class predictions
    class_predictions_thresholded = (class_predictions >= 0.5).astype(int)

    # Calculate accuracy for this class
    class_accuracy = accuracy_score(true_labels, class_predictions_thresholded)

    # Store the accuracy for this class label
    class_accuracies[class_label] = class_accuracy

# Print the accuracy for each class label
for class_label, accuracy in class_accuracies.items():
    print(f"Accuracy for class {class_label}: {accuracy:.2f}")
if isinstance(y_disorder_test, pd.Series):
    y_disorder_test = y_disorder_test.to_numpy()
n_classes = 8
fpr_disorder = dict()
tpr_disorder = dict()
roc_auc_disorder = dict()

for i in range(n_classes):
    fpr_disorder[i], tpr_disorder[i], _ = roc_curve(y_disorder_test_bin[:, i], y_disorder_scores_test[:, i])
    roc_auc_disorder[i] = auc(fpr_disorder[i], tpr_disorder[i])

# Plot ROC curve for each class
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr_disorder[i], tpr_disorder[i], label='ROC Curve ({} vs Rest) (AUC = {:.2f})'.format(class_labels[i], roc_auc_disorder[i]))

plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Disorder Prediction)')
plt.legend()
plt.show()
# Assuming you have the true labels stored in y_true
y_true_result = y_result_test  # Assuming this is how you obtain true labels for the 'Result' prediction
y_true_disorder = y_disorder_test  # Assuming this is how you obtain true labels for the 'disorder' prediction

# Assuming you have calculated the predicted probabilities for each class
y_proba_result = y_result_scores_test  # Replace this with your actual predicted probabilities for 'Result' prediction
y_proba_disorder = y_disorder_scores_test  # Replace this with your actual predicted probabilities for 'disorder' prediction

# Check shapes of true labels and predicted probabilities
print("Shapes of true labels and predicted probabilities for 'Result' prediction:")
print("True labels shape:", y_true_result.shape)
print("Predicted probabilities shape:", y_proba_result.shape)

print("\nShapes of true labels and predicted probabilities for 'disorder' prediction:")
print("True labels shape:", y_true_disorder.shape)
print("Predicted probabilities shape:", y_proba_disorder.shape)
y_true_disorder_bin = label_binarize(y_true_disorder, classes=np.unique(y_true_disorder))
# Initialize dictionaries to store precision, recall, and thresholds for each class
precision_dict = {}
recall_dict = {}
thresholds_dict = {}

# Iterate over each class
for class_label in range(y_proba_disorder.shape[1]):
    # Compute precision, recall, and thresholds for the current class
    precision, recall, thresholds = precision_recall_curve(y_true_disorder == class_label, y_proba_disorder[:, class_label])
    
    # Store precision, recall, and thresholds for the current class
    precision_dict[class_label] = precision
    recall_dict[class_label] = recall
    thresholds_dict[class_label] = thresholds

# Choose the predicted probabilities for the positive class in the 'disorder' prediction
# You need to determine the index of the positive class in the 'disorder' prediction
positive_class_index_disorder = 0  # Adjust this index accordingly based on your class encoding

precisions = []
recalls = []

# Iterate over each class
for class_index in range(y_proba_disorder.shape[1]):
    # Compute precision, recall, and thresholds for the current class
    precision, recall, _ = precision_recall_curve(
        y_true_disorder_bin[:, class_index], y_proba_disorder[:, class_index]
    )
    
    # Append precision and recall values to lists
    precisions.append(precision)
    recalls.append(recall)

# Plot Precision-Recall curve for each class
plt.figure(figsize=(10, 6))
for class_index in range(y_proba_disorder.shape[1]):
    plt.plot(recalls[class_index], precisions[class_index], marker='.', label=f'Disorder {class_index}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Disorders')
plt.legend()
plt.grid(True)
plt.show()

# Initialize a list to store accuracies for each disorder
accuracies = []

# Iterate over each class
for class_index in range(y_proba_disorder.shape[1]):
    # Predicted class labels are the class with the highest predicted probability
    predicted_labels = np.argmax(y_proba_disorder, axis=1)
    
    # True class labels
    true_labels = y_true_disorder
    
    # Filter true and predicted labels for the current class
    true_labels_class = (true_labels == class_index)
    predicted_labels_class = (predicted_labels == class_index)
    
    # Compute accuracy for the current class
    accuracy = accuracy_score(true_labels_class, predicted_labels_class)
    
    # Append accuracy to the list
    accuracies.append(accuracy)
# Print accuracies for each disorder
for class_index, accuracy in enumerate(accuracies):
    print(f"Accuracy for Disorder {class_index}: {accuracy:.4f}")
# Plot the distribution of accuracies for each disorder
class_distribution = y_result_train.value_counts()
print("Class Distribution:\n", class_distribution)
# Apply SMOTE resampling to address class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_result_train)
# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_result_train), y=y_result_train)
class_weight_dict = dict(zip(np.unique(y_result_train), class_weights))
# Instantiate PolynomialFeatures to create interaction terms up to degree 2
poly = PolynomialFeatures(degree=2, interaction_only=True)

# Transform the features
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
# Print the shape of the transformed feature matrices
print("Shape of X_train_poly:", X_train_poly.shape)
print("Shape of X_test_poly:", X_test_poly.shape)

# Alternatively, you can inspect the first few rows of the transformed feature matrices
print("Sample data from X_train_poly:")
print(X_train_poly[:5])  # Print the first 5 rows
print("Sample data from X_test_poly:")
print(X_test_poly[:5])   # Print the first 5 rows
joblib.dump(mlp_classifier, 'scikit_model.joblib')
# Choose the optimal threshold based on the ROC curve or other metrics
# For example, you can choose the threshold that maximizes the F1-score
# Calculate F1-score for each threshold
f1_scores = [f1_score(y_result_test, y_proba_result >= threshold) for threshold in thresholds]

# Find the threshold that maximizes the F1-score
optimal_threshold_index = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_threshold_index]
optimal_f1_score = f1_scores[optimal_threshold_index]
print("Optimal Threshold (Maximizing F1-score):", optimal_threshold)
print("F1-score at Optimal Threshold:", optimal_f1_score)




















