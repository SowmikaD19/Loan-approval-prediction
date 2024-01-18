import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
df = pd.read_csv('train.csv')

# Preprocess the data
df['self_employed'] = df['self_employed'].replace({' No': 0, ' Yes': 1})
df['education'] = df['education'].replace({' Graduate': 1, ' Not Graduate': 0})
df['loan_status'] = df['loan_status'].replace({' Approved': 1, ' Rejected': 0})

# Split the data
X = df.drop(["loan_id", "loan_status"], axis=1)
y = df["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Print training and test scores
print(f"Training Score: {model.score(X_train, y_train) * 100}%")
print(f"Test score: {model.score(X_test, y_test) * 100}%")

# Save the trained model to a pickle file
with open('loan_approval_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Print a message indicating that the model has been saved
print("Trained model saved to loan_approval_model.pkl")