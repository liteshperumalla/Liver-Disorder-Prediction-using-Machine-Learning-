import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# Load the saved model
loaded_model = joblib.load('/Users/liteshperumalla/Desktop/Files/projects/scikit_model.joblib')

# Load the new dataset
new_data = pd.read_csv('/Users/liteshperumalla/Desktop/Files/projects/archive/test.csv', encoding='unicode_escape')

# Preprocess the new dataset
# Make sure to perform the same preprocessing steps as done during training
# For example:
# Handle missing values
new_data = new_data.bfill()
# Encode categorical variables
new_data['Gender'] = new_data['Gender'].astype('category').cat.codes
# Scale numerical features
# Assuming you have already applied min-max scaling during training
# You may need to apply the same scaling to the new dataset


# Make predictions
# Assuming X_new contains the features of the new dataset
X_new = new_data 
numeric_columns = X_new.select_dtypes(include=['float64', 'int64']).columns

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Apply min-max scaling to numerical features
X_new[numeric_columns] = scaler.fit_transform(X_new[numeric_columns])

# Now, X_new contains scaled numerical features
print(X_new.head())
# Predict 'Result' (presence or absence of liver disorder)
result_predictions = loaded_model.predict(X_new)
# Predict 'disorder' (type of disorder)
disorder_predictions = loaded_model.predict_disorder(X_new)

# Assuming you want to add the predictions to the new dataset
new_data['Result_Predictions'] = result_predictions
new_data['Disorder_Predictions'] = disorder_predictions

# Save the updated dataset with predictions
new_data.to_csv('/path/to/your/new_dataset_with_predictions.csv', index=False)
