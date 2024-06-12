import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("test.csv")
base=data["TotalBsmtSF"]#base sqfoot
bathroom=data["BsmtFullBath"]#no of bathrooms
bedroom=data["BedroomAbvGr"]# no  of bedrooms
sns.pairplot(data)#https://github.com/huzaifsayed/Linear-Regression-Model-for-House-Price-Prediction/blob/master/Linear%20Regression%20Model.ipynb
