import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

# Load the CSV file into a pandas DataFrame
csv_file = "databreaches650.csv"
df = pd.read_csv(csv_file)

#Reset the row indices of the DataFrame
df = df.reset_index(drop=True) 

# Define a function to preprocess the text data
stemmer = PorterStemmer()

# Define a function to preprocess the text data and extract the year
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Parse and clean HTML
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    # Extract the year using regular expressions
    year = re.findall(r'\b\d{4}\b', text)
    # Remove the year from the text
    text = re.sub(r'\b\d{4}\b', '', text)
    # Remove non-alphabetic characters and digits
    text = re.sub(r'\W+', ' ', text)
    # Tokenize the text into words
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    # Stem the words
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    # Rejoin the words into a single string
    text = ' '.join(words)
    # Create a new column for the year extracted from the text
    year = year[0] if year else ''
    return pd.Series([text, year])

# Apply the preprocess_text function to the Description column
df[['Description', 'Year']] = df['Description'].apply(preprocess_text)

# Convert the text data into numerical features using the TF-IDF representation
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Description'])

# Define the category labels for each cluster
labels = ['Retail', 'Healthcare', 'Financial', 'Other']

# Cluster the data using the K-Means algorithm
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# Define the number of clusters
num_clusters = 6

# Perform clustering using KMeans
km = KMeans(n_clusters=num_clusters, random_state=42)
km.fit(X.toarray())

# Predict the categories of the data points using the K-Means algorithm
y_pred = kmeans.predict(X)

# Assign a label to each data point based on the cluster it belongs to
df['Category'] = [labels[i] for i in y_pred]

#preprocess the 'Cluster' and 'Category' column 
df['Cluster'] = kmeans.labels_
df['Category']= df['Category'].apply(lambda x: re.sub(r'\W+', '', x))

# Assign cluster labels to each row
df['Cluster'] = km.labels_

# Define a function to map industries to categories
def map_industry_to_category(industry):
    if industry == 'Government':
        return 'Government'
    elif industry == 'Education':
        return 'Education'
    elif industry == 'Social Media':
        return 'Social Media'
    else:
        return 'Other'

# Map industries to categories in the Category column
df['Category'] = df['Category'].apply(map_industry_to_category)

#Refine the 'Other' category into more specific industries based on the characteristics of the Description column
other_df = df[df['Category'] == 'Other']
other_text = other_df['Description'].values
other_labels = ['Government', 'Education', 'Healthcare', 'Financial', 'Social Media', 'Retail', 'Other']
other_kmeans = KMeans(n_clusters=len(other_labels), random_state=42)
other_X = vectorizer.transform(other_text)
other_kmeans.fit(other_X)
other_y_pred = other_kmeans.predict(other_X)
other_df['Industry'] = [other_labels[i] for i in other_y_pred]

# Merge the Industry column back into the original DataFrame
df = pd.merge(df, other_df[['Industry']], left_index=True, right_index=True, how='outer')
df['Industry'] = df['Industry'].fillna('N/A')

# Define the color map for the industries
colors = {'Government': 'red', 'Education': 'green', 'Healthcare': 'blue',
          'Financial': 'orange', 'Social Media': 'purple', 'Retail': 'brown', 'N/A': 'gray'}

# Visualize the clusters and industries using PCA and a scatter plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

try:
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=[colors[x] for x in df['Industry']], cmap='Set2')
except KeyError:
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=[colors.get(x, 'gray') for x in df['Industry']], cmap='Set2')

plt.title('Clusters and Industries Visualization')
plt.show()

# Count the number of data points for each industry within each cluster
grouped = df.groupby(['Cluster','Industry']).size().reset_index(name='Count')

# Create a pivot table to count the number of data points for each industry within each cluster
pivot = grouped.pivot(index='Cluster', columns='Industry', values='Count')

# Visualize the pivot table as a heatmap
sns.heatmap(pivot, annot=True, cmap='Blues')
plt.title('Number of Data Points for Each Industry within Each Cluster')
plt.show()


# Compute the correlation matrix
corr_matrix = df.corr()

# Create a correlation heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# Show the plot
plt.show()

print(df.head())
print(df.info())

# Group the data by Year, Category, and Industry, and sum the PwnCount for each group
grouped = df.groupby(['Year', 'Category', 'Industry'])['PwnCount'].sum()

# Convert the grouped data to a DataFrame and reset the index
summary = pd.DataFrame(grouped).reset_index()

# Display the summary table
print(summary)

# Group the data by Year and Industry, and sum the PwnCount for each group
grouped = df.groupby(['Year', 'Industry'])['PwnCount'].sum()

# Convert the grouped data to a DataFrame and reset the index
summary = pd.DataFrame(grouped).reset_index()

# Pivot the data to create a matrix of PwnCount values for each Subcategory and Year
pivoted = summary.pivot(index='Year', columns='Industry', values='PwnCount')

# Visualize the data using a stacked bar chart
pivoted.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.xlabel('Year')
plt.ylabel('PwnCount')
plt.title('PwnCount by Industry and Year')
plt.show()

# Plot the clusters on a scatter plot
sns.scatterplot(x='PwnCount', y='Year', hue='Cluster', data=df, palette='Set1')
plt.xlabel('PwnCount')
plt.ylabel('Year')
plt.title('Data Breach Clusters')
plt.show()

# Compute the mean PwnCount and Year for each cluster
summary = df.groupby('Cluster').agg({'PwnCount': 'mean', 'Year': 'mean'})

# Print the summary statistics for each cluster
print(summary)

# Compute the frequency of each DataClasses within each cluster
dataclass_counts = df.groupby('Cluster')['DataClasses'].apply(lambda x: x.value_counts(normalize=True))

# Print the most common DataClasses in each cluster
print(dataclass_counts)

# Group the data by Year and compute the sum of PwnCount for each year
yearly_pwn_count = df.groupby('Year')['PwnCount'].sum().reset_index()

# Apply a log transform to the PwnCount values
yearly_pwn_count['log_PwnCount'] = np.log(yearly_pwn_count['PwnCount'])

# Create a line plot of log_PwnCount vs Year
sns.lineplot(x='Year', y='log_PwnCount', data=yearly_pwn_count)

# Set the plot title and axis labels
plt.title('Data Breaches by Year')
plt.xlabel('Year')
plt.ylabel('log(PwnCount)')

# Show the plot
plt.show()

# Group the data by Name and compute the sum of PwnCount for each Name
name_pwn_count = df.groupby('Name')['PwnCount'].sum().reset_index()

# Filter the data to include only the top 10 data breaches by PwnCount
top_10 = name_pwn_count.nlargest(10, 'PwnCount')

# Merge the top 10 data breaches back into the original DataFrame
top_10_df = pd.merge(df, top_10, on='Name')

# Group the top 10 data breaches by Year and compute the sum of PwnCount for each year
yearly_pwn_count = top_10_df.groupby('Year')['PwnCount_y'].sum().reset_index()

# Create a line plot of PwnCount vs Year for the top 10 data breaches
sns.lineplot(x='Year', y='PwnCount_y', hue='Name', data=top_10_df)

# Set the plot title and axis labels
plt.title('Top 10 Data Breaches by Year')
plt.xlabel('Year')
plt.ylabel('PwnCount')

# Show the plot
plt.show()