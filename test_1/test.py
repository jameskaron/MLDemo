# importing pandas package
import pandas as pd

# making data frame from csv file
data = pd.read_csv("nba.csv")

# retrieving rows by loc method
# row1 = data.loc[3]
# row1 = data.iloc[0][['Team', 'Position']]
row1 = data.iloc[0][['Team', 'Position']]
# retrieving rows by iloc method
# row2 = data.iloc[3]
row2 = data.iloc[20][['Team', 'Position']]
# checking if values are equal
row1 == row2

print("row1: \n", row1)
print("row2: \n", row2)