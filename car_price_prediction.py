import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("car_data.csv")

# Select features and target
X = data[['Year', 'Present_Price', 'Kms_Driven']]
y = data['Selling_Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
print("Model trained successfully!")
print("R² Score:", model.score(X_test, y_test))
