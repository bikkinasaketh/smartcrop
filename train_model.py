import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier


# Load dataset
data = pd.read_csv("dataset/crop_recommendation.csv")

print("Dataset Loaded Successfully")
print(data.head())


# Features and Label
X = data.drop("label", axis=1)
y = data["label"]


# Encode labels (important)
encoder = LabelEncoder()
y = encoder.fit_transform(y)


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# Train model
model = XGBClassifier()
model.fit(X_train, y_train)

print("Model Training Completed")


# Prediction
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")


# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Save encoder also
pickle.dump(encoder, open("encoder.pkl", "wb"))

print("Model and Encoder saved successfully")