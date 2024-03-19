import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data_path = 'Fish.csv'  # Make sure to replace this with the correct path
fish_data = pd.read_csv(data_path)

# Separating the features and the target variable
X = fish_data.drop('Species', axis=1)  # Features
y = fish_data['Species']  # Target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the evaluation results
print(f"Accuracy: {accuracy*100:.2f}%")
print("Classification Report:")
print(classification_rep)

import pickle

# Assuming 'classifier' is your trained model
model = rf_classifier

# Save the model to disk
filename = 'vijay_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved to {filename}")
