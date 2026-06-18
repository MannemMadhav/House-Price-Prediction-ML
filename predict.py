import pandas as pd
import joblib

model = joblib.load("house_price_model.pkl")

area = int(input("Enter Area: "))
bedrooms = int(input("Enter Bedrooms: "))

new_house = pd.DataFrame({
    "Area": [area],
    "Bedrooms": [bedrooms]
})

prediction = model.predict(new_house)

print("Predicted Price =", prediction[0])