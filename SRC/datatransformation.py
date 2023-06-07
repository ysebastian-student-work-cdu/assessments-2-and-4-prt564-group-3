import pandas as pd
import ast
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib
import textwrap
import numpy as array
import numpy as mean
import numpy as cov
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency
import datetime

csv_file = "databreaches650.csv"
df = pd.read_csv(csv_file)
df = df.fillna("")
print(df.head())
print(df.info())

# Extract the column you want to analyze for outliers
column_to_analyze = df['PwnCount']

# Define a function to detect outliers using the z-score method
def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    return np.where(z_scores > threshold)

# Detect outliers using the z-score method
outliers = detect_outliers_zscore(column_to_analyze)

# Print the indices of the outliers
print(outliers)

# Compute the correlation matrix
corr_matrix = df.corr()

# Create a correlation heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# Show the plot
plt.show()

#convert string data in column 'Dataclasses' to a list of Python objects using ast.literal_eval()
df['Categorized Data Class'] = df['DataClasses'].apply(lambda x: ast.literal_eval(x))

#create am empty dictionary 'dct' to count the frequency of each element in 'Categorized Data Class' acress all rows
dct = {}
for idx, row in df.iterrows():
    # iterate over each element in 'Categorized Data Class' for the current row
    for ele in row["Categorized Data Class"]:
        # If the element is not already present in 'dct', add it and set its count to 1
        if ele not in dct:
            dct[ele] = 1
        # If the element is already present in 'dct', increment its count by 1
        else:
            dct[ele] += 1

#print the dictionary containing the frequency count of each element in 'Categorized Data Class' across all rows
print(dct)
#Filter the dictionary to only include elements with a count of 10 or greater
filtered_dct = {k: v for k, v in dct.items() if v >= 10}

# Create a bar chart
plt.bar(filtered_dct.keys(), filtered_dct.values())

plt.xticks(rotation=90)
plt.gca().set_xticklabels([textwrap.fill(label,10) for label in filtered_dct.keys()])
# Add labels and title
plt.xlabel('Breach data type')
plt.ylabel('Count')
plt.title('Count of each breach data class')

plt.show()

# Convert boolean variables to categorical variables
df['IsVerified'] = df['IsVerified'].astype('category')
df['IsFabricated'] = df['IsFabricated'].astype('category')
df['IsSensitive'] = df['IsSensitive'].astype('category')
df['IsRetired'] = df['IsRetired'].astype('category')
df['IsSpamList'] = df['IsSpamList'].astype('category')
df['IsMalware'] = df['IsMalware'].astype('category')

# Count the occurrences of each category in the categorical variables
counts = df[['IsVerified', 'IsFabricated', 'IsSensitive', 'IsRetired', 'IsSpamList', 'IsMalware']].apply(pd.Series.value_counts)

# Plot the category counts in a stacked bar chart
counts.plot(kind='bar', stacked=True)
plt.xlabel('Type of breach')
plt.ylabel('Count')
plt.title('Category Counts in Type of breach')

plt.show()

# Convert date columns of 'BreachDate', 'AddedDate', and 'ModifiedDate' from object format to datetime format
df['BreachDate'] = pd.to_datetime(df['BreachDate'])
df['AddedDate'] = pd.to_datetime(df['AddedDate'])
df['ModifiedDate'] = pd.to_datetime(df['ModifiedDate'])

# create a new column 'weekday' containing the weekday name
df['weekday'] = df['BreachDate'].dt.strftime('%A')

# create a new column 'month' containing the month name
df['month'] = df['BreachDate'].dt.strftime('%B')

# count the frequency of each weekday and month
weekday_counts = df['weekday'].value_counts()
month_counts = df['month'].value_counts()

# create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# plot the weekday counts in the first subplot
weekday_counts.plot(kind='bar', ax=ax1)
ax1.set_title('Frequency of BreachDate in Weekdays')
ax1.set_xlabel('Weekday')
ax1.set_ylabel('Frequency')

# plot the month counts in the second subplot
month_counts.plot(kind='bar', ax=ax2)
ax2.set_title('Frequency of BreachDate in Months')
ax2.set_xlabel('Month')
ax2.set_ylabel('Frequency')

# adjust the layout and display the figure
plt.tight_layout()
plt.show()

# group the DataFrame by year and count the number of breaches for each year
year_counts = df.groupby(df['BreachDate'].dt.year)['BreachDate'].count()

# create a bar chart of the year counts
plt.bar(year_counts.index, year_counts.values)

# add labels and title
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.title('Frequency of Breaches by Year')

# set the x-tick positions and labels to show each individual year
plt.xticks(range(2007, 2024), rotation=90)

# display the plot
plt.show()

# create a figure with 6 subplots arranged in a 3x2 grid
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 9))

# iterate over the years from 2017 to 2022
for i, year in enumerate(range(2017, 2023)):

    # calculate the subplot row and column index from the iteration index
    row = i // 2
    col = i % 2

    # filter the DataFrame to include only the rows for the current year
    df_year = df[df['BreachDate'].dt.year == year]

    # count the frequency of breaches within the current year subset
    month_counts = df_year['BreachDate'].dt.month.value_counts()

    # create a bar chart of the month counts for the current year
    axes[row, col].bar(month_counts.index, month_counts.values)

    # add labels and title to the current subplot
    axes[row, col].set_xlabel('Month')
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].set_title(f'Frequency of Breaches in {year} by Month')

# adjust the layout of the subplots
fig.tight_layout()

# display the plot
plt.show()

# Create seasonal indicators based on the month of the BreachDate column
df['BreachMonth'] = df['BreachDate'].dt.month
df['BreachIsWinter'] = (df['BreachMonth'] == 12) | (df['BreachMonth'] <= 2)
df['BreachIsSpring'] = (df['BreachMonth'] >= 3) & (df['BreachMonth'] <= 5)
df['BreachIsSummer'] = (df['BreachMonth'] >= 6) & (df['BreachMonth'] <= 8)
df['BreachIsFall'] = (df['BreachMonth'] >= 9) & (df['BreachMonth'] <= 11)

# Group data by season and count the number of data breaches
season_counts = df.groupby(['BreachIsWinter', 'BreachIsSpring', 'BreachIsSummer', 'BreachIsFall']).size()

# Create a bar chart of the season counts
season_counts.plot(kind='bar')
plt.title('Number of Data Breaches by Season')
plt.xlabel('Season')
plt.ylabel('Number of Data Breaches')
plt.xticks([0, 1, 2, 3], ['Winter', 'Spring', 'Summer', 'Fall'], rotation=0)
plt.show()

# compute the time difference in days
df['time_diff'] = (df['AddedDate'] - df['BreachDate']).dt.days

# print summary statistics
print(df.describe())

# create a histogram with custom bin edges
bin_edges = np.arange(0, 100, 5) # 5-day bins from 0 to 100 days
plt.hist(df['time_diff'], bins=bin_edges)

# add labels and title
plt.xlabel('Time Difference (Days)')
plt.ylabel('Frequency')
plt.title('Histogram of Time Differences between BreachDate and AddedDate')

# set the x-tick labels to show the bin edges
plt.xticks(bin_edges)

# show the plot
plt.show()

from sklearn.preprocessing import MinMaxScaler

# create a MinMaxScaler object
scaler = MinMaxScaler()

# fit and transform the time_diff column
df['time_diff_scaled'] = scaler.fit_transform(df[['time_diff']])

# create a box plot of the original and transformed data
df[['time_diff', 'time_diff_scaled']].plot(kind='box')
plt.title('MinMaxScaler test of time difference between breachdate and addeddate')
# show the plot
plt.show()

from sklearn.preprocessing import StandardScaler

# create a StandardScaler object
scaler = StandardScaler()

# fit and transform the time_diff column
df['time_diff_scaled'] = scaler.fit_transform(df[['time_diff']])

# create a histogram of the original and transformed data
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].hist(df['time_diff'], bins=25)
ax[0].set_xlabel('Time Difference (Days)')
ax[0].set_ylabel('Frequency')
ax[0].set_title('Histogram of Time Differences (Original Data)')
ax[1].hist(df['time_diff_scaled'], bins=25)
ax[1].set_xlabel('Time Difference (Scaled)')
ax[1].set_ylabel('Frequency')
ax[1].set_title('Histogram of Time Differences (Standardized Data)')

# show the plot
plt.show()

from sklearn.decomposition import PCA

# create a PCA object
pca = PCA(n_components=2)

# fit and transform the time_diff column
df_pca = pca.fit_transform(df[['time_diff', 'PwnCount']])

# create a scatter plot of the transformed data
plt.scatter(df_pca[:, 0], df_pca[:, 1])
plt.xlabel('Time Difference')
plt.ylabel('PwnCount')
plt.title('PCA of Time Differences and PwnCount')

# show the plot
plt.show()

# extract the PwnCount column as a NumPy array
pwn_count = df['PwnCount'].values

# apply min-max scaling to the PwnCount column
pwn_count_minmax = (pwn_count - np.min(pwn_count)) / (np.max(pwn_count) - np.min(pwn_count))

# create a histogram of the scaled PwnCount values
plt.hist(pwn_count_minmax, bins=30, alpha=0.5, color='blue')
plt.xlabel('Scaled PwnCount')
plt.ylabel('Frequency')
plt.title('Histogram of Scaled PwnCount Values')
plt.show()

print(df.dtypes)
