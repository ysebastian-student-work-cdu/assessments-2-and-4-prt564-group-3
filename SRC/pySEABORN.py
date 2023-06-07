import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# You can import custom data
df = pd.read_csv('out.csv')


#---we not using this BEGIN
# Seaborn provides built in datasets
###print(sns.get_dataset_names())
# Load a built in dataset based on US State car crash percentages
###crash_df = sns.load_dataset('car_crashes')
#---we not using this END


# Jointplot compares 2 distributions and plots a scatter plot by default
# As we can see as people tend to speed they also tend to drink & drive
# With kind you can create a regression line with kind='reg'
# You can create a 2D KDE with kind='kde'
# Kernal Density Estimation estimates the distribution of data
# You can create a hexagon distribution with kind='hex'
sns.jointplot(x='Subcategory_code', y='PwnCount', data=df, kind='reg')

# Get just the KDE plot
sns.kdeplot(df['PwnCount'])

# Pair Plot plots relationships across the entire data frames numerical values
sns.pairplot(df)
#---we not using this END
# Load data on tips
###tips_df = sns.load_dataset('tips')

# With hue you can pass in a categorical column and the charts will be colorized
# You can use color maps from Matplotlib to define what colors to use
# sns.pairplot(tips_df, hue='sex', palette='Blues')

# Plots a single column of datapoints in an array as sticks on an axis
# With a rug plot you'll see a more dense number of lines where the amount is
# most common. This is like how a histogram is taller where values are more common
##sns.rugplot(df['PwnCount']) ##NOT WORKING !!

# You can set styling for your axes and grids
# white, darkgrid, whitegrid, dark, ticks
sns.set_style('white')

# You can use figure sizing from Matplotlib
plt.figure(figsize=(8,4))

# Change size of lables, lines and other elements to best fit
# how you will present your data (paper, talk, poster)
##sns.set_context('PwnCount', font_scale=1.4)
##sns.jointplot(x='PwnCount', y='Subcategory_code', data=df, kind='reg')

# Get rid of spines
# You can turn of specific spines with right=True, left=True
# bottom=True, top=True
##sns.despine(left=False, bottom=False)

# Focus on distributions using categorical data in reference to one of the numerical
# columns

# Aggregate categorical data based on a function (mean is the default)
# Estimate total bill amount based on sex
# With estimator you can define functions to use other than the mean like those
# provided by NumPy : median, std, var, cov or make your own functions
sns.barplot(x='PwnCount',y='Subcategory_code',data=df, estimator=np.median)

# A count plot is like a bar plot, but the estimator is counting
# the number of occurances
sns.countplot(x='PwnCount',data=df)

plt.figure(figsize=(14,9))
sns.set_style('darkgrid')

# A box plot allows you to compare different variables
# The box shows the quartiles of the data. The bar in the middle is the median and
# the box extends 1 standard deviation from the median
# The whiskers extend to all the other data aside from the points that are considered
# to be outliers
# Hue can add another category being sex
# We see men spend way more on Friday versus less than women on Saturday
sns.boxplot(x='Subcategory_code',y='PwnCount',data=df, hue='IsVerified')

# Moves legend to the best position
plt.legend(loc=0)

# Violin Plot is a combination of the boxplot and KDE
# While a box plot corresponds to data points, the violin plot uses the KDE estimation
# of the data points
# Split allows you to compare how the categories compare to each other
sns.violinplot(x='Subcategory_code',y='PwnCount',data=df, hue='IsVerified',split=True)

plt.figure(figsize=(8,5))

# The strip plot draws a scatter plot representing all data points where one
# variable is categorical. It is often used to show all observations with
# a box plot that represents the average distribution
# Jitter spreads data points out so that they aren't stacked on top of each other
# Hue breaks data into men and women
# Dodge separates the men and women data
sns.stripplot(x='Subcategory_code',y='PwnCount',data=df, jitter=True,
              hue='IsVerified', dodge=True)


##t is like a strip plot, but points are adjusted so they don't overlap
# It looks like a combination of the violin and strip plots
# sns.swarmplot(x='day',y='total_bill',data=tips_df)

# You can stack a violin plot with a swarm
sns.violinplot(x='Subcategory_code',y='PwnCount',data=df)
sns.swarmplot(x='Subcategory_code',y='PwnCount',data=df, color='white')

plt.figure(figsize=(8,6))

sns.set_style('dark')

sns.set_context('talk')
plt.show()
exit()

#-------------------------ENDDDD REST STARTS BREAKING
# You can use Matplotlibs color maps for color styling
# https://matplotlib.org/3.3.1/tutorials/colors/colormaps.html
sns.stripplot(x='Subcategory_code',
              y='PwnCount',data=df,
              hue='IsVerified',
              palette='seismic')

# Add the optional legend with a location number (best: 0,
# upper right: 1, upper left: 2, lower left: 3, lower right: 4,
# https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.legend.html)
# or supply a tuple of x & y from lower left
plt.legend(loc=0)

plt.figure(figsize=(8,6))
sns.set_context('PwnCount', font_scale=1.4)


# To create a heatmap with data you must have data set up as a matrix where variables
# are on the columns and rows

# Correlation tells you how influential a variable is on the result
# So we see that n previous accident is heavily correlated with accidents, while the
# insurance premium is not
####crash_mx = crash_df.corr()

# Create the heatmap, add annotations and a color map
sns.heatmap(df, annot=True, cmap='Blues')

plt.figure(figsize=(8,6))
sns.set_context('PwnCount', font_scale=1.4)

# We can create a matrix with an index of month, columns representing years
# and the number of passengers for each
# We see that flights have increased over time and that most people travel in
# July and August
#####flights = sns.load_dataset("flights")
#####flights = flights.pivot_table(index='month', columns='year', values='passengers')
# You can separate data with lines
sns.heatmap(df, cmap='Blues', linecolor='white', linewidth=1)


plt.figure(figsize=(8,6))
sns.set_context('PwnCount', font_scale=1.4)

# A Cluster map is a hierarchically clustered heatmap
# The distance between points is calculated, the closest are joined, and this
# continues for the next closest (It compares columns / rows of the heatmap)
# This is data on iris flowers with data on petal lengths
####iris = sns.load_dataset("iris")
# Return values for species
# species = iris.pop("species")
# sns.clustermap(iris)

# With our flights data we can see that years have been reoriented to place
# like data closer together
# You can see clusters of data for July & August for the years 59 & 60
# standard_scale normalizes the data to focus on the clustering
sns.clustermap(df,cmap="Blues", standard_scale=1)

plt.figure(figsize=(8,6))
sns.set_context('PwnCount', font_scale=1.4)

# You can create a grid of different plots with complete control over what is displayed
# Create the empty grid system using the provided data
# Colorize based on species
# iris_g = sns.PairGrid(iris, hue="species")

# Put a scatter plot across the upper, lower and diagonal
# iris_g.map(plt.scatter)

# Put a histogram on the diagonal
# iris_g.map_diag(plt.hist)
# And a scatter plot every place else
# iris_g.map_offdiag(plt.scatter)

# Have different plots in upper, lower and diagonal
# iris_g.map_upper(plt.scatter)
# iris_g.map_lower(sns.kdeplot)

# You can define define variables for x & y for a custom grid
iris_g = sns.PairGrid(df, hue="Subcategory_code",
                      x_vars=["Subcategory_code", "IsSensitive"],
                      y_vars=["PwnCount", "IsVerified"])

iris_g.map(plt.scatter)

# Add a legend last
iris_g.add_legend()

# Can also print multiple plots in a grid in which you define columns & rows
# Get histogram for smokers and non with total bill for lunch & dinner
# tips_fg = sns.FacetGrid(tips_df, col='time', row='smoker')

# You can pass in attributes for the histogram
# tips_fg.map(plt.hist, "total_bill", bins=8)

# Create a scatter plot with data on total bill & tip (You need to parameters)
# tips_fg.map(plt.scatter, "total_bill", "tip")

# We can assign variables to different colors and increase size of grid
# Aspect is 1.3 x the size of height
# You can change the order of the columns
# Define the palette used
# tips_fg = sns.FacetGrid(tips_df, col='time', hue='smoker', height=4, aspect=1.3,
#                       col_order=['Dinner', 'Lunch'], palette='Set1')
# tips_fg.map(plt.scatter, "total_bill", "tip", edgecolor='w')

# # Define size, linewidth and assign a color of white to markers
# kws = dict(s=50, linewidth=.5, edgecolor="w")
# # Define that we want to assign different markers to smokers and non
# tips_fg = sns.FacetGrid(tips_df, col='sex', hue='smoker', height=4, aspect=1.3,
#                         hue_order=['Yes','No'],
#                         hue_kws=dict(marker=['^', 'v']))
# tips_fg.map(plt.scatter, "total_bill", "tip", **kws)
# tips_fg.add_legend()

# This dataframe provides scores for different students based on the level
# of attention they could provide during testing
#####att_df = sns.load_dataset("attention")
# Put each person in their own plot with 5 per line and plot their scores
att_fg = sns.FacetGrid(df, col='Subcategory_code', col_wrap=5, height=1.5)
att_fg.map(plt.plot, 'Subcategory_code', 'PwnCount', marker='.')

# lmplot combines regression plots with facet grid
###tips_df = sns.load_dataset('tips')
###df.head()

plt.figure(figsize=(8,6))
sns.set_context('PwnCount', font_scale=1.4)

plt.figure(figsize=(8,6))

# We can plot a regression plot studying whether total bill effects the tip
# hue is used to show separation based off of categorical data
# We see that males tend to tip slightly more
# Define different markers for men and women
# You can effect the scatter plot by passing in a dictionary for styling of markers
sns.lmplot(x='Subcategory_code', y='PwnCount', hue='IsSensitive', data=df, markers=['o', '^'],
          scatter_kws={'s': 100, 'linewidth': 0.5, 'edgecolor': 'w'})

# You can separate the data into separate columns for day data
# sns.lmplot(x='total_bill', y='tip', col='sex', row='time', data=tips_df)
#####tips_df.head()

# Makes the fonts more readable
sns.set_context('poster', font_scale=1.4)

sns.lmplot(x='Subcategory_code',
           y='PwnCount',
           data=df,
           col='day',
           hue='IsSensitive',
          height=8, aspect=0.6)

