import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("test.csv")
price_data = pd.read_csv("sample_submission.csv")
X = data[["TotalBsmtSF", "BsmtFullBath", "BedroomAbvGr"]]
z = X.fillna(X.mean())
y = price_data["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(z, y, test_size=0.4, random_state=101)
lm = LinearRegression()
lm.fit(X_train, y_train)

# Test case:
basement_in_sqft = 1329
bathrooms = 0
bedrooms = 3
da = np.array([[basement_in_sqft, bathrooms, bedrooms]])

# Reshape da with feature names
da = pd.DataFrame(da, columns=["TotalBsmtSF", "BsmtFullBath", "BedroomAbvGr"])

# Predict using the model
predictions = lm.predict(da)
print(f"Predicted price is {predictions[0]}")
