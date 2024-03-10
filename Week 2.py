import pandas as pd
import numpy as np

# 1) Extracting 'Manufacturer', 'Model', and 'Type' for every 20th row starting from 1st
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
result_1 = df[['Manufacturer', 'Model', 'Type']].iloc[::20, :]

# 2) Replacing missing values in Min.Price and Max.Price columns with their respective mean
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
df['Min.Price'] = df['Min.Price'].fillna(df['Min.Price'].mean())
df['Max.Price'] = df['Max.Price'].fillna(df['Max.Price'].mean())

# 3) Getting rows of a dataframe with row sum > 100
df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))
rows_sum_gt_100 = df[df.sum(axis=1) > 100]

# 4) Creating a 4x4 NumPy array filled with random integers between 1 and 100
arr = np.random.randint(1, 100, size=(4, 4))

# Reshape separate 2D arrays for rows and columns
row_array = arr.reshape(2, 8)
col_array = arr.T.reshape(2, 8)
# Calculating sum of each row and each column separately using lambda function
sum_row = np.apply_along_axis(lambda x: np.sum(x), axis=1, arr=row_array)
sum_col = np.apply_along_axis(lambda x: np.sum(x), axis=1, arr=col_array)
print("Result 1:\n", result_1)
print("\nResult 2:\n", df)
print("\nResult 3:\n", rows_sum_gt_100)
print("\nRow Sums:\n", sum_row)
print("\nColumn Sums:\n", sum_col)
