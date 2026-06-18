import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("dataset/kc_house_data.csv")

# Features
X = df[["bedrooms", "bathrooms", "sqft_living"]]

# Target
y = df["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
score = r2_score(y_test, y_pred)

print("Model Accuracy (R² Score):", score)

# Save model
joblib.dump(model, "house_price_model.pkl")

print("Model Saved Successfully!")