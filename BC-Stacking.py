import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

# Load dataset
file_path = r"C:\Users\Azooo\OneDrive\سطح المكتب\Cns\data.csv"
data = pd.read_csv(file_path)

# Target column is the column we want to predict for the test data
target_column = 'diagnosis'

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = data.drop(target_column, axis=1)
X = imputer.fit_transform(X)
y = data[target_column]

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Define base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=50)),
    ('lr', LogisticRegression(max_iter=100, random_state=50)),
    ('knn', KNeighborsClassifier(n_neighbors=3))
]

# Define meta-model
meta_model = LogisticRegression(max_iter=100, random_state=50)

# Create the stacking classifier
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

start_time = time.time()

# Train the stacking model
stacking_model.fit(X_train, y_train)
end_time = time.time()
training_time = end_time - start_time

start_time = time.time()
y_pred = stacking_model.predict(X_test)
end_time = time.time()
prediction_time = end_time - start_time

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')
print(f'Training Time: {training_time} seconds')
print(f'Prediction Time: {prediction_time} seconds')

# Implement cross-validation
cv_scores = cross_val_score(stacking_model, X, y, cv=3, scoring='accuracy')

print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean Cross-Validation Score: {np.mean(cv_scores)}')
