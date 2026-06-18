import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

data = {
    "Area": [1000, 1500, 2000, 2500, 3000],
    "Bedrooms": [2, 3, 3, 4, 4],
    "Price": [20, 30, 40, 50, 60]
}

df = pd.DataFrame(data)

X = df[["Area", "Bedrooms"]]
y = df["Price"]

model = LinearRegression()

model.fit(X, y)
joblib.dump(model, "house_price_model.pkl")

print("Model Saved Successfully!")

area = int(input("Enter Area: "))
bedrooms = int(input("Enter Bedrooms: "))

new_house = pd.DataFrame({
    "Area": [area],
    "Bedrooms": [bedrooms]
})

prediction = model.predict(new_house)

print("Predicted Price =", prediction[0])