import numpy as np
import pandas as pd

# Notes taken from https://pandas.pydata.org/pandas-docs/stable/10min.html

# row-based definition of frame using a List
df = pd.DataFrame(
    [
        [0, "zero", "repeating value", pd.Timestamp("20000101")],
        [1, "one", "repeating value", pd.Timestamp("20000102")],
        [2, "two", "repeating value", pd.Timestamp("20000103")],
        [3, "three", "repeating value", pd.Timestamp("20000104")]
    ],
    index = ["x0", "x1", "x2", "x3"],
    columns = ["A", "B", "C", "D" ] 
)

# The above dataframe is identical to the below one

# column-based definition of frame using a Dictionary
df = pd.DataFrame( 
    { 
        "A": range(4), 
        "B": [ "zero", "one", "two", "three"], 
        "C": "repeating value",
        "D": pd.date_range("20000101", periods=4)
    },
    index = ["x0", "x1", "x2", "x3"],
)


# Cool things we can do
df.index
df.columns
df.values
df.info() # types, size
df.describe() # min/max, quantiles, mean, std
df.T # transpose
df.sort_index(axis = 1, ascending = False) # note we have a 1-axis index
df.sort_values(by = "B")
df2 = df.copy()
df.drop(columns=["A","B"], axis=1) # this returns a copy of the table - does not change it in place

# In all cases, selection will result in a dynically choosen type depending on what is being returned. It will be one of the following:
#  - Scalar
#  - Series
#  - DataFrame

# SIMPLE SELECTION BY COLUMNS OR ROWS
# -----------------------------------

df.B            # A column, returned as a Series
df["B"]         # A column, returned as a Series
df[0:2]         # A row filter (index positions, EXCLUSIVE), returned as DataFrame
df["x0":"x2"]   # A row filter (index values, INCLUSIVE), returned as DataFrame

# SINGLE AXIS LABEL SELECTION
# ---------------------------

df.loc[ "x0" ]          # A single row cross section identified by index LABEL value, as Series
df.loc[ ["x0","x2"] ]   # Multiple row cross section identified by index LABEL values list, as DataFrame

# MULTI AXIS LABEL SELECTION
# --------------------------

df.loc[:, ["A", "B"]]           # all rows, reducing dimensions as DataFrame
df.loc["x0":"x2", ["A", "B"]]   # subset rows, reducing dimensions as DataFrame

# SCALAR LABEL SELECTION
# ----------------------

df.loc["x0", "A"]  # NOOOOOO
df.at[ "x0", "A"]  # <--- MUCH FASTER - USE .at[] instead of 

# SELECTING BY POSITION

df.iloc[ 1 ]            # A single row cross section identified by index INDEX POSITION, as Series
df.iloc[ 1:2, 0:3 ]     # Subset rows, reducing dimensions, indentified by POSITION, as DataFrame
df.iat[  1, 1 ]         # Fast scalar access by POSITION

# BOOLEAN INDEXING

df.loc[ [ True, True, False, False] ]   # Multiple row cross section identified by boolean list (length == nrows), as DataFrame
df.loc[ df["A"] < 2 ]                   # Same idea, but use expression to produce boolean list
df.loc[ df["A"].isin([0, 1]) ]          # Same idea, excpet introducing the .isin() function

# SETTING DATA

df2 = df.copy()
pd.isna(df2)            # Boolean mask for values that are NA (e.g. None or numpy.NaN)
pd.isna(df2).sum()      # Count of total number of nas across each column
df2.dropna(how="any")   # Drops any row with missing data
df2.fillna(value=5)     # Sets missing data

minPerColumn = df.apply(lambda x: min(x)) # Operates one column at a time