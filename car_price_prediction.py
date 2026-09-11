# Car Price Prediction using Machine Learning
# Author: Ankush Kumar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("data/f.csv")

# EDA
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

sns.histplot(df['price'], bins=50, kde=True)
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

sns.boxplot(data=df, x='year', y='price')
plt.xticks(rotation=90)
plt.show()

sns.scatterplot(data=df, x='mileage', y='price')
plt.show()

sns.boxplot(data=df, x='engineSize', y='price')
plt.show()

sns.boxplot(data=df, x='transmission', y='price')
plt.show()

sns.boxplot(data=df, x='fuelType', y='price')
plt.show()

sns.boxplot(data=df, x='model', y='price')
plt.xticks(rotation=90)
plt.show()

# Feature / target split
x = df.drop(columns=['price'], axis=1)
y = df['price']

# One-hot encoding
x_encoded = pd.get_dummies(
    x,
    columns=['model', 'transmission', 'fuelType'],
    drop_first=True,
    dtype=int
)

# Feature scaling
numerical_columns = ['year', 'mileage', 'tax', 'mpg']
scaler = StandardScaler()
x_encoded[numerical_columns] = scaler.fit_transform(x_encoded[numerical_columns])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x_encoded, y, test_size=0.33, random_state=42
)

# Linear Regression
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

n = x_test.shape[0]
p = x_test.shape[1]
adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

print(f"R2 Score: {r2:.4f}")
print(f"Adjusted R2 Score: {adjusted_r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
