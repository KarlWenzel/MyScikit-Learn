import numpy as np
np.random.seed(42)  # we may or may not need a seed, but it's a good practice for reproducibility

# pandas tutorial - https://pandas.pydata.org/pandas-docs/stable/10min.html
# pandas cheatsheet - http://datacamp-community.s3.amazonaws.com/9f0f2ae1-8bd8-4302-a67b-e17f3059d9e8
import pandas as pd
from pandas.plotting import scatter_matrix

# jupyter notebook supports inline plotting 
import matplotlib
import matplotlib.pyplot as plt

# http://scikit-learn.org/stable/modules/classes.html
from sklearn import model_selection, linear_model
from sklearn.metrics import mean_squared_error


# download from https://www.kaggle.com/mirichoi0218/insurance/version/1
dataFile = "C:\\Users\\zkew18d\\source\\repos\\MyScikit-Learn\\data\\insurance.csv"

# read the data and output some basic descriptive info
rawData = pd.read_csv(dataFile)
rawData.head()

