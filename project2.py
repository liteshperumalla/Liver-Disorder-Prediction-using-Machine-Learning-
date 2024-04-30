import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load new data
new_data = pd.read_csv('/Users/liteshperumalla/Desktop/Files/projects/archive/test.csv', encoding='unicode_escape')

# Check and preprocess new data
print(new_data.info())
print(new_data.isnull().sum())
print('Gender' in new_data.columns)
new_data['Gender'] = new_data['Gender'].astype('category').cat.codes
imputer = SimpleImputer(strategy='mean')
new_data = pd.DataFrame(imputer.fit_transform(new_data), columns=new_data.columns)
scaler = StandardScaler()
new_data_scaled = pd.DataFrame(scaler.fit_transform(new_data), columns=new_data.columns)

# Load the pre-trained model
model = joblib.load('/Users/liteshperumalla/Desktop/Files/projects/scikit_model.joblib')

# Make predictions
y_pred = model.predict(new_data_scaled)
print('Predicted Labels:\n', y_pred)
