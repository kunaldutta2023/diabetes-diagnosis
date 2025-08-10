import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv('diabetes.csv')  # Ensure this file is present

# Features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
print('\nModel Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---- Add user input function and prediction ----

# Get the list of feature names in order
feature_names = X.columns.tolist()

def get_user_input():
    print("\nEnter the following values to predict diabetes risk:")
    user_data = []
    for feature in feature_names:
        while True:
            try:
                value = float(input(f"{feature}: "))
                user_data.append(value)
                break
            except ValueError:
                print(f"Invalid input. Please enter a number for {feature}.")
    return np.array(user_data).reshape(1, -1)

# Take input, scale, predict
user_input = get_user_input()
user_input_scaled = scaler.transform(user_input)
user_pred = model.predict(user_input_scaled)

print("\nPrediction result:")
if user_pred[0] == 1:
    print("The model predicts: DIABETES PRESENT (high risk).\n")
else:
    print("The model predicts: NO DIABETES (low risk).\n")
